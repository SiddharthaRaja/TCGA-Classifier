#!/usr/bin/env python3
"""
Tile each patient's slides once and push the patches through all backbones.

Sequential and resumable. Only touches patients the download script has marked
complete, and only deletes a slide once every requested backbone has saved
features for it.

Usage
    python extract_all_backbones.py                        # process everything, sequentially
    python extract_all_backbones.py --limit 2 --keep-wsi   # smoke test, keeps the slides
    python extract_all_backbones.py --time-budget-hours 11
    python extract_all_backbones.py --backbones densenet121,convnext
    python extract_all_backbones.py --status
    python extract_all_backbones.py --selftest             # no GPU/slides needed

What changed vs the previous version, and why
---------------------------------------------
* GATED ON THE DOWNLOAD MARKER. A patient is only considered once
  download_complete.json exists AND every WSI file it lists is present at the
  recorded size. Previously the gate was "is there any .svs here", so a patient
  whose second slide was still transferring got extracted from slide 1 alone
  and then marked done forever.

* MAGNIFICATION IS NOW REAL. get_best_level_for_downsample(2) returns level 0
  on typical TCGA slides (their levels are 1, 4, 16, 32), so the old code read
  256px at native 40x while believing it was reading 20x. Patients scanned at
  40x got half the tissue context of those scanned at 20x — a confound baked
  straight into the feature space. We now read step_l0 = 256 * (base_mag/20)
  level-0 pixels and downsample to 256, so every patch covers the same physical
  area regardless of how the slide was scanned.

* TISSUE IS FOUND ON A THUMBNAIL FIRST. The old code called read_region on
  every candidate coordinate just to discover most were white background —
  roughly 90,000 full-resolution reads per slide. We threshold a single
  thumbnail, then only read coordinates that pass. Typically an 3-5x cut in I/O.

* MEMORY IS BOUNDED. Patches used to accumulate in one list for the whole
  patient (~5GB of PIL images before a single forward pass). We now stream in
  chunks: read a chunk, run all backbones over it, keep only the feature rows.

* ONE BACKBONE CANNOT KILL THE RUN. Each backbone is wrapped separately, and
  per-backbone .npz files mean a failure only costs that backbone for that
  patient. Re-running fills in whatever is missing without redoing the rest —
  which is also how you add UNI later without recomputing anything.

* CTransPath gets resized to 224. It is a Swin built at img_size=224/window 7;
  feeding it 256 produces a 64x64 grid that does not divide by the window size.

* THE SLIDE IS DELETED LAST, AND ONLY ON FULL SUCCESS. A thumbnail is saved to
  $HOME state first, so attention heatmaps remain possible after the slide is
  gone.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import tcga_common as C

# ---------------------------------------------------------------- config
PATCH_SIZE = 256          # the size the backbones actually see
TARGET_MAG = 20.0         # every patch covers the same physical area at this mag
TISSUE_THRESHOLD = 0.10   # fraction of non-white pixels needed to keep a patch
GRAY_CUTOFF = 220         # below this grey value counts as tissue
CHUNK_SIZE = 256          # patches held in memory at once
BATCH_SIZE = 64           # forward-pass batch
THUMB_MAX = 4096          # tissue-mask thumbnail, also saved for heatmaps

ALL_BACKBONES = ["densenet121", "convnext", "phikon", "ctranspath", "retccl"]
EXPECTED_DIMS = {"densenet121": 1024, "convnext": 768, "phikon": 768,
                 "ctranspath": 768, "retccl": 2048}


def _lazy_torch():
    import torch
    return torch


# ------------------------------------------------------------------ models
def load_backbones(names, device):
    """Load once, keep resident, so one tiling pass feeds every backbone.

    Each entry is (model, transform_fn, dim). transform_fn takes a PIL image at
    PATCH_SIZE and returns a CHW tensor — this is where per-backbone input size
    differences live, rather than assuming one transform suits all five.
    """
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    imagenet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # CTransPath is a Swin built at 224 with window 7. 256 gives a 64x64 grid,
    # which does not divide by 7. Resize rather than hope.
    imagenet_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    loaded = {}

    if "densenet121" in names:
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()
        loaded["densenet121"] = (m.to(device).eval(), imagenet, 1024)

    if "convnext" in names:
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Keep the LayerNorm2d + Flatten from the original head and drop only
        # the final Linear. Replacing the whole head with a bare Flatten (as
        # before) silently skipped the normalisation the features expect.
        m.classifier = nn.Sequential(m.classifier[0], m.classifier[1], nn.Identity())
        loaded["convnext"] = (m.to(device).eval(), imagenet, 768)

    if "phikon" in names:
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        phikon = AutoModel.from_pretrained("owkin/phikon").to(device).eval()

        def phikon_transform(pil_img):
            return processor(pil_img, return_tensors="pt")["pixel_values"][0]

        class PhikonWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(pixel_values=x).last_hidden_state[:, 0, :]

        loaded["phikon"] = (PhikonWrapper(phikon).to(device).eval(), phikon_transform, 768)

    if "ctranspath" in names:
        import timm
        try:
            from timm.layers import to_2tuple
        except ImportError:
            from timm.models.layers.helpers import to_2tuple

        class ConvStem(nn.Module):
            """Matches TransPath's own ConvStem forward (channels-last permute,
            not flatten+transpose), which is what the converted hub weights
            below expect."""

            def __init__(self, img_size=224, patch_size=4, in_chans=3,
                         embed_dim=768, norm_layer=None, **kwargs):
                super().__init__()
                assert patch_size == 4
                assert embed_dim % 8 == 0
                img_size = to_2tuple(img_size)
                patch_size = to_2tuple(patch_size)
                self.img_size = img_size
                self.patch_size = patch_size
                self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
                self.num_patches = self.grid_size[0] * self.grid_size[1]
                stem = []
                input_dim, output_dim = 3, embed_dim // 8
                for _ in range(2):
                    stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3,
                                          stride=2, padding=1, bias=False))
                    stem.append(nn.BatchNorm2d(output_dim))
                    stem.append(nn.ReLU(inplace=True))
                    input_dim = output_dim
                    output_dim *= 2
                stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
                self.proj = nn.Sequential(*stem)
                self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

            def forward(self, x):
                x = self.proj(x)
                x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
                return self.norm(x)

        ctp = timm.create_model(
            "hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
            embed_layer=ConvStem, pretrained=True, num_classes=0)
        loaded["ctranspath"] = (ctp.to(device).eval(), imagenet_224, 768)

    if "retccl" in names:
        from huggingface_hub import hf_hub_download
        w = hf_hub_download(repo_id="jamesdolezal/RetCCL",
                            filename="retccl_torchscript.pth")
        m = torch.jit.load(w, map_location=device)
        m.eval()
        loaded["retccl"] = (m, imagenet, 2048)

    return loaded


# ------------------------------------------------------------------ tiling
def slide_base_magnification(slide) -> float:
    """objective-power when present, else derive from microns-per-pixel
    (0.25 um/px is 40x, 0.5 is 20x). Falls back to 20x rather than guessing
    high, so a missing property costs context rather than corrupting scale."""
    props = slide.properties
    obj = props.get("openslide.objective-power")
    if obj:
        try:
            return float(obj)
        except (TypeError, ValueError):
            pass
    mpp = props.get("openslide.mpp-x")
    if mpp:
        try:
            return 10.0 / float(mpp)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return 20.0


def tissue_mask(slide, thumb_max=THUMB_MAX):
    """One thumbnail, thresholded once, reused for every candidate coordinate.

    Returns (mask, scale_x, scale_y, thumb). Replaces reading tens of thousands
    of full-resolution regions just to find out they were background.
    """
    w0, h0 = slide.level_dimensions[0]
    thumb = slide.get_thumbnail((thumb_max, thumb_max)).convert("RGB")
    arr = np.asarray(thumb).astype(np.float32)
    mask = arr.mean(axis=2) < GRAY_CUTOFF
    tw, th = thumb.size
    return mask, tw / w0, th / h0, thumb


def plan_patches(slide):
    """Decide the read geometry, then keep only coordinates with tissue.

    step_l0 is the level-0 footprint of one output patch. At 40x base mag that
    is 512 level-0 pixels downsampled to 256 — the same physical area a 20x
    slide covers with 256 pixels. That equivalence is the whole point.
    """
    base_mag = slide_base_magnification(slide)
    downsample_needed = max(1.0, base_mag / TARGET_MAG)
    level = slide.get_best_level_for_downsample(downsample_needed)
    level_ds = slide.level_downsamples[level]
    read_size = max(1, int(round(PATCH_SIZE * downsample_needed / level_ds)))
    step_l0 = max(1, int(round(PATCH_SIZE * downsample_needed)))

    mask, sx, sy, thumb = tissue_mask(slide)
    mh, mw = mask.shape
    w0, h0 = slide.level_dimensions[0]

    coords = []
    for y in range(0, h0 - step_l0 + 1, step_l0):
        y0 = int(y * sy)
        y1 = max(y0 + 1, int((y + step_l0) * sy))
        if y0 >= mh:
            continue
        row = mask[y0:min(y1, mh)]
        if not row.any():
            continue  # whole strip is background
        for x in range(0, w0 - step_l0 + 1, step_l0):
            x0 = int(x * sx)
            x1 = max(x0 + 1, int((x + step_l0) * sx))
            if x0 >= mw:
                continue
            box = row[:, x0:min(x1, mw)]
            if box.size and box.mean() > TISSUE_THRESHOLD:
                coords.append((x, y))

    return {"coords": coords, "level": level, "read_size": read_size,
            "step_l0": step_l0, "base_mag": base_mag, "thumb": thumb}


def iter_patch_chunks(svs_paths, chunk_size=CHUNK_SIZE):
    """Yield (coords_chunk, images_chunk) with bounded memory.

    Patches from every slide a patient has are merged into one bag, and each
    coordinate carries its slide index so a patch stays traceable to its source
    file. Some TCGA-BRCA patients have two diagnostic slides; dropping the
    second silently discards real tissue.
    """
    import openslide
    plans = []
    for svs in svs_paths:
        slide = openslide.OpenSlide(str(svs))
        plan = plan_patches(slide)
        plans.append((slide, plan))

    buf_c, buf_i = [], []
    try:
        for slide_idx, (slide, plan) in enumerate(plans):
            for (x, y) in plan["coords"]:
                try:
                    region = slide.read_region((x, y), plan["level"],
                                               (plan["read_size"], plan["read_size"]))
                    region = region.convert("RGB")
                except Exception:
                    continue
                if plan["read_size"] != PATCH_SIZE:
                    region = region.resize((PATCH_SIZE, PATCH_SIZE))
                buf_c.append((slide_idx, x, y))
                buf_i.append(region)
                if len(buf_i) >= chunk_size:
                    yield buf_c, buf_i
                    buf_c, buf_i = [], []
        if buf_i:
            yield buf_c, buf_i
    finally:
        for slide, _ in plans:
            slide.close()


def plans_metadata(svs_paths):
    """Geometry only, for the npz header — heatmaps need step_l0 to draw a
    patch at the right size, which the old code got wrong by assuming 256."""
    import openslide
    meta = []
    for svs in svs_paths:
        slide = openslide.OpenSlide(str(svs))
        try:
            plan = plan_patches(slide)
            meta.append({"file": str(svs), "level": plan["level"],
                         "step_l0": plan["step_l0"], "base_mag": plan["base_mag"],
                         "n_coords": len(plan["coords"])})
        finally:
            slide.close()
    return meta


# ------------------------------------------------------------- inference
def run_chunk(model, transform_fn, images, device, use_amp):
    torch = _lazy_torch()
    feats = []
    for i in range(0, len(images), BATCH_SIZE):
        batch = [transform_fn(im) for im in images[i:i + BATCH_SIZE]]
        t = torch.stack(batch).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(t)
            else:
                out = model(t)
        feats.append(out.float().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else None


def save_features(path: Path, features, coords, slide_files, geom, dtype):
    """Write to .tmp then rename — a killed job never leaves a half-written npz
    that a later run would mistake for a finished one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # NOTE: np.savez_compressed appends ".npz" to a *path* that lacks it, which
    # would write to "<name>.npz.tmp.npz" and leave the rename below pointing at
    # nothing. Handing it an open file object bypasses that entirely.
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            features=features.astype(dtype),
            coords=coords,
            slide_files=np.array([str(s) for s in slide_files]),
            geometry=np.array([json.dumps(geom)]),
            patch_size=np.array([PATCH_SIZE]),
            target_mag=np.array([TARGET_MAG]),
        )
    os.replace(tmp, path)


# ------------------------------------------------------------------ gating
def wsi_files_for(pid, marker):
    """The slides the download marker promised, resolved to real paths.

    Returns (paths, missing). A non-empty missing list means the raw data was
    lost after download — the marker is then cleared so the download job
    re-fetches, rather than this job quietly extracting a partial patient.
    """
    entries = (marker.get("files") or {}).get("wsi") or []
    paths, missing = [], []
    for e in entries:
        p = C.RAW_DIR / pid / "WSI" / e["name"]
        if not p.exists():
            missing.append(e["name"])
        elif e.get("size") and p.stat().st_size != e["size"]:
            missing.append(f"{e['name']} (wrong size)")
        else:
            paths.append(p)
    return paths, missing


def status(cases, backbones):
    ready = extracted = no_wsi = 0
    per_bb = {bb: 0 for bb in backbones}
    for c in cases:
        pid = c["submitter_id"]
        m = C.read_marker(pid, C.DOWNLOAD_MARKER)
        if not m:
            continue
        if not (m.get("files") or {}).get("wsi"):
            no_wsi += 1
            continue
        ready += 1
        if C.is_extracted(pid):
            extracted += 1
        for bb in backbones:
            if (C.FEATURES_PATCH_ROOT / bb / f"{pid}.npz").exists():
                per_bb[bb] += 1
    print(f"  downloaded with WSI : {ready}")
    print(f"  fully extracted     : {extracted}")
    print(f"  downloaded, no WSI  : {no_wsi}  (tabular only — nothing to extract)")
    for bb in backbones:
        print(f"    {bb:<14}: {per_bb[bb]} npz files")


# ---------------------------------------------------------------- selftest
def selftest():
    """Exercises the geometry and gating logic with a fake slide — no GPU, no
    openslide, no downloads. Run this first on a new node to catch a broken
    environment before burning queue time."""
    class FakeSlide:
        def __init__(self, w, h, obj, downsamples):
            self.level_dimensions = [(w // int(d), h // int(d)) for d in downsamples]
            self.level_downsamples = list(downsamples)
            self.properties = {"openslide.objective-power": obj} if obj else {}

        def get_best_level_for_downsample(self, ds):
            best = 0
            for i, d in enumerate(self.level_downsamples):
                if d <= ds:
                    best = i
            return best

    ok = True
    for obj, expect_step in [("40", 512), ("20", 256)]:
        s = FakeSlide(100000, 60000, obj, [1, 4, 16, 32])
        base = slide_base_magnification(s)
        need = max(1.0, base / TARGET_MAG)
        lvl = s.get_best_level_for_downsample(need)
        step = int(round(PATCH_SIZE * need))
        read = int(round(PATCH_SIZE * need / s.level_downsamples[lvl]))
        good = (step == expect_step)
        ok &= good
        print(f"  base={base:>4.0f}x -> level={lvl} read_size={read} step_l0={step} "
              f"{'OK' if good else 'FAIL'}")
    s = FakeSlide(1000, 1000, None, [1])
    s.properties = {"openslide.mpp-x": "0.25"}
    m = slide_base_magnification(s)
    print(f"  mpp 0.25 -> {m:.0f}x {'OK' if abs(m - 40) < 0.01 else 'FAIL'}")
    ok &= abs(m - 40) < 0.01
    print("\n  selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="check geometry logic, no data needed")
    ap.add_argument("--limit", type=int, default=None, help="stop after N patients")
    ap.add_argument("--time-budget-hours", type=float, default=None,
                    help="stop starting new patients after this long")
    ap.add_argument("--backbones", type=str, default=",".join(ALL_BACKBONES))
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                    help="patch feature storage precision (float16 halves disk use)")
    ap.add_argument("--keep-wsi", action="store_true",
                    help="do not delete slides after extraction (use for test runs)")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1,
                    help="optional stable manifest-based split; default 1 = sequential")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    unknown = [b for b in backbones if b not in ALL_BACKBONES]
    if unknown:
        raise SystemExit(f"unknown backbone(s): {unknown}; known: {ALL_BACKBONES}")

    C.ensure_dirs()
    C.banner("Tiling + multi-backbone feature extraction")
    C.describe_roots()

    cases = C.load_manifest()
    if args.status:
        status(cases, backbones)
        return

    torch = _lazy_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and not args.no_amp
    dtype = np.float16 if args.dtype == "float16" else np.float32
    print(f"  device={device}  amp={use_amp}  dtype={args.dtype}")
    print(f"  backbones: {backbones}", flush=True)

    mine = C.select_patients(cases, args.shard, args.n_shards)
    ctl = C.RunController(args.time_budget_hours)

    models_loaded = None
    done = skipped = failed = 0

    for i, case in enumerate(mine, 1):
        pid = case["submitter_id"]

        stop, why = ctl.should_stop()
        if stop:
            print(f"\n  stopping: {why}. Resubmit to continue.", flush=True)
            break
        if args.limit and done >= args.limit:
            print(f"\n  reached --limit {args.limit}.", flush=True)
            break

        if C.is_extracted(pid):
            skipped += 1
            continue
        marker = C.read_marker(pid, C.DOWNLOAD_MARKER)
        if marker is None:
            continue  # not downloaded yet — the download job owns this patient
        if not (marker.get("files") or {}).get("wsi"):
            # Genuinely has no diagnostic slide. Mark it so we stop revisiting.
            C.write_marker(pid, C.EXTRACT_MARKER,
                           {"patient_id": pid, "backbones": {}, "n_patches": 0,
                            "note": "no diagnostic slide available"})
            skipped += 1
            continue

        svs_paths, missing = wsi_files_for(pid, marker)
        if missing:
            print(f"  [{i}] {pid} — raw slides missing/corrupt {missing}; "
                  f"clearing download marker so they get re-fetched", flush=True)
            C.clear_marker(pid, C.DOWNLOAD_MARKER)
            failed += 1
            continue

        todo = [bb for bb in backbones
                if not (C.FEATURES_PATCH_ROOT / bb / f"{pid}.npz").exists()]
        if not todo:
            C.write_marker(pid, C.EXTRACT_MARKER, {
                "patient_id": pid,
                "backbones": {bb: str(C.FEATURES_PATCH_ROOT / bb / f"{pid}.npz")
                              for bb in backbones},
            })
            if not args.keep_wsi:
                for f in svs_paths:
                    f.unlink(missing_ok=True)
            skipped += 1
            continue

        with C.PatientLock(pid) as lk:
            if not lk.acquired:
                print(f"  [{i}] {pid} — locked by another job, skipping", flush=True)
                continue

            if models_loaded is None:
                print(f"  loading {len(backbones)} backbones onto {device} ...", flush=True)
                models_loaded = load_backbones(backbones, device)
                print("  done loading.\n", flush=True)

            t0 = time.time()
            try:
                geom = plans_metadata(svs_paths)
                n_expected = sum(g["n_coords"] for g in geom)
                if n_expected == 0:
                    print(f"  [{i}] {pid} — 0 tissue patches, marking done", flush=True)
                    C.write_marker(pid, C.EXTRACT_MARKER,
                                   {"patient_id": pid, "backbones": {}, "n_patches": 0,
                                    "note": "no tissue detected"})
                    if not args.keep_wsi:
                        for f in svs_paths:
                            f.unlink(missing_ok=True)
                    continue

                # Save the thumbnail BEFORE the slide is eligible for deletion,
                # so heatmaps stay possible once the .svs is gone.
                import openslide
                s0 = openslide.OpenSlide(str(svs_paths[0]))
                try:
                    C.THUMBS_DIR.mkdir(parents=True, exist_ok=True)
                    s0.get_thumbnail((2048, 2048)).convert("RGB").save(
                        C.THUMBS_DIR / f"{pid}.jpg", quality=85)
                finally:
                    s0.close()

                # One tiling pass; every backbone consumes the same chunk.
                acc = {bb: [] for bb in todo}
                all_coords = []
                failures = {}
                for coords_chunk, images_chunk in iter_patch_chunks(svs_paths):
                    all_coords.extend(coords_chunk)
                    for bb in list(acc.keys()):
                        model, tfm, dim = models_loaded[bb]
                        try:
                            f = run_chunk(model, tfm, images_chunk, device, use_amp)
                        except Exception as e:
                            failures[bb] = repr(e)
                            acc.pop(bb, None)
                            print(f"      {bb} FAILED: {e}", flush=True)
                            continue
                        acc[bb].append(f)

                coords_arr = np.array(all_coords, dtype=np.int64)
                results = {}
                for bb, parts in acc.items():
                    feats = np.concatenate(parts, axis=0)
                    if feats.shape[0] != coords_arr.shape[0]:
                        failures[bb] = f"row mismatch {feats.shape[0]} vs {coords_arr.shape[0]}"
                        continue
                    if feats.shape[1] != EXPECTED_DIMS[bb]:
                        failures[bb] = f"dim {feats.shape[1]} != expected {EXPECTED_DIMS[bb]}"
                        continue
                    save_features(C.FEATURES_PATCH_ROOT / bb / f"{pid}.npz",
                                  feats, coords_arr, svs_paths, geom, dtype)
                    pooled_dir = C.FEATURES_POOLED_ROOT / bb
                    pooled_dir.mkdir(parents=True, exist_ok=True)
                    tmp = pooled_dir / f"{pid}.npz.tmp"
                    with open(tmp, "wb") as fh:
                        np.savez_compressed(fh, pooled=feats.mean(axis=0).astype(np.float32))
                    os.replace(tmp, pooled_dir / f"{pid}.npz")
                    results[bb] = "ok"

                have_all = all((C.FEATURES_PATCH_ROOT / bb / f"{pid}.npz").exists()
                               for bb in backbones)
                n_slides = f" ({len(svs_paths)} slides)" if len(svs_paths) > 1 else ""
                print(f"  [{i}] {pid} — {coords_arr.shape[0]} patches{n_slides} — "
                      f"{results}{' FAILED:' + str(failures) if failures else ''} — "
                      f"{time.time() - t0:.1f}s", flush=True)

                if have_all:
                    C.write_marker(pid, C.EXTRACT_MARKER, {
                        "patient_id": pid,
                        "n_patches": int(coords_arr.shape[0]),
                        "n_slides": len(svs_paths),
                        "dtype": args.dtype,
                        "backbones": {bb: EXPECTED_DIMS[bb] for bb in backbones},
                        "geometry": geom,
                    })
                    # Only now is the slide expendable: marker written, features
                    # on disk, thumbnail saved.
                    if not args.keep_wsi:
                        for f in svs_paths:
                            f.unlink(missing_ok=True)
                    done += 1
                else:
                    print("      incomplete — keeping slides, will retry next run",
                          flush=True)
                    failed += 1

            except Exception as e:
                print(f"  [{i}] {pid} — PATIENT FAILED: {e}", flush=True)
                failed += 1

    print(f"\n  extracted: {done}   already done: {skipped}   failed (will retry): {failed}")
    print(f"  elapsed: {ctl.elapsed_h():.2f}h")
    print("  run  python extract_all_backbones.py --status  for the overall picture.",
          flush=True)


if __name__ == "__main__":
    main()
