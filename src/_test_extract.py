#!/usr/bin/env python3
"""Exercises extract_all_backbones.py's gating, resume and deletion logic with
a synthetic slide. Stubs the GPU backbones (the maths is not what has been
failing); everything about *which patient gets touched when* is real code."""
import json, os, shutil, sys, tempfile, types
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

root = Path(tempfile.mkdtemp(prefix="tcga_x_"))
os.environ["TCGA_DATA_ROOT"] = str(root / "data")
os.environ["TCGA_STATE_ROOT"] = str(root / "state")

# ---- fake openslide -------------------------------------------------------
class FakeSlide:
    def __init__(self, path):
        self.path = str(path)
        self.img = Image.new("RGB", (2000, 1500), (245, 245, 245))
        for x in range(200, 1400):          # a dark "tissue" band
            for y in range(200, 1000, 7):
                self.img.putpixel((x, y), (120, 80, 130))
        self.level_dimensions = [(2000, 1500)]
        self.level_downsamples = [1.0]
        self.properties = {"openslide.objective-power": "40"}

    def get_best_level_for_downsample(self, ds): return 0
    def read_region(self, loc, level, size):
        return self.img.crop((loc[0], loc[1], loc[0] + size[0], loc[1] + size[1])).convert("RGBA")
    def get_thumbnail(self, size):
        t = self.img.copy(); t.thumbnail(size); return t
    def close(self): pass

fake_os_mod = types.ModuleType("openslide")
fake_os_mod.OpenSlide = FakeSlide
sys.modules["openslide"] = fake_os_mod

import tcga_common as C
import extract_all_backbones as X

# ---- stub the heavy bits --------------------------------------------------
fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
X._lazy_torch = lambda: fake_torch
X.load_backbones = lambda names, device: {n: (None, None, X.EXPECTED_DIMS[n]) for n in names}

FAIL_BACKBONE = {"name": None}

def load_with_names(names, device):
    # the "model" slot carries the backbone name so the stub below knows
    # which one it is standing in for
    return {n: (n, None, X.EXPECTED_DIMS[n]) for n in names}
X.load_backbones = load_with_names

def run_chunk_stub(model, tfm, images, device, use_amp):
    if FAIL_BACKBONE["name"] == model:
        raise RuntimeError("simulated backbone failure")
    return np.random.default_rng(0).normal(
        size=(len(images), X.EXPECTED_DIMS[model])).astype(np.float32)
X.run_chunk = run_chunk_stub

def patched_main(argv):
    sys.argv = ["extract_all_backbones.py"] + argv
    X.main()

# ---- set up patients ------------------------------------------------------
C.ensure_dirs()
BB = ["densenet121", "convnext"]
cases = [{"case_id": f"c{i}", "submitter_id": f"TCGA-ZZ-{i:04d}"} for i in range(4)]
C.MANIFEST.write_text(json.dumps(cases))

def make_patient(pid, n_slides, marked=True, sizes_ok=True):
    wsi = []
    d = C.RAW_DIR / pid / "WSI"; d.mkdir(parents=True, exist_ok=True)
    for k in range(n_slides):
        f = d / f"{pid}-{k}.svs"
        f.write_bytes(b"x" * 1000)
        wsi.append({"name": f.name, "size": 1000 if sizes_ok else 999})
    if marked:
        C.write_marker(pid, C.DOWNLOAD_MARKER, {"patient_id": pid, "files": {"wsi": wsi}})

make_patient("TCGA-ZZ-0000", 1)
make_patient("TCGA-ZZ-0001", 2)                       # two diagnostic slides
C.write_marker("TCGA-ZZ-0002", C.DOWNLOAD_MARKER,
               {"patient_id": "TCGA-ZZ-0002", "files": {"wsi": []}})   # no slide
make_patient("TCGA-ZZ-0003", 1, marked=False)          # downloading, no marker yet

fails = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail and not cond else ''}")
    if not cond: fails.append(name)

print("\n=== 1. first extraction pass ===")
patched_main(["--backbones", ",".join(BB)])

for pid in ("TCGA-ZZ-0000", "TCGA-ZZ-0001"):
    check(f"{pid} marked extracted", C.is_extracted(pid))
    for bb in BB:
        check(f"{pid} has {bb} npz", (C.FEATURES_PATCH_ROOT / bb / f"{pid}.npz").exists())
    check(f"{pid} slides deleted", not list((C.RAW_DIR / pid / "WSI").glob("*.svs")))
    check(f"{pid} thumbnail saved for later heatmaps",
          (C.THUMBS_DIR / f"{pid}.jpg").exists())

d = np.load(C.FEATURES_PATCH_ROOT / "densenet121" / "TCGA-ZZ-0001.npz")
check("two-slide patient merges both slides into one bag",
      len(set(d["coords"][:, 0].tolist())) == 2, str(set(d["coords"][:, 0].tolist())))
check("coords carry slide index + x + y", d["coords"].shape[1] == 3)
check("features stored as float16 by default", d["features"].dtype == np.float16,
      str(d["features"].dtype))
check("rows match coords", d["features"].shape[0] == d["coords"].shape[0])
geom = json.loads(str(d["geometry"][0]))
check("geometry records 512px level-0 step for a 40x slide",
      geom[0]["step_l0"] == 512, str(geom[0]))

check("patient with no diagnostic slide is marked done, not retried forever",
      C.is_extracted("TCGA-ZZ-0002"))
check("un-marked patient (still downloading) is left alone",
      not C.is_extracted("TCGA-ZZ-0003"))
check("un-marked patient's slide NOT consumed",
      list((C.RAW_DIR / "TCGA-ZZ-0003" / "WSI").glob("*.svs")) != [])

print("\n=== 2. rerun does no repeated work ===")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    patched_main(["--backbones", ",".join(BB)])
out = buf.getvalue()
check("second pass extracts nothing new", "extracted: 0" in out, out[-300:])

print("\n=== 3. raw data lost after download -> marker cleared, not half-extracted ===")
pid = "TCGA-ZZ-0003"
C.write_marker(pid, C.DOWNLOAD_MARKER,
               {"patient_id": pid, "files": {"wsi": [{"name": f"{pid}-0.svs", "size": 1000}]}})
for f in (C.RAW_DIR / pid / "WSI").glob("*.svs"):
    f.unlink()                                   # simulate a scratch wipe
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    patched_main(["--backbones", ",".join(BB)])
out = buf.getvalue()
check("missing raw slide clears the download marker", not C.is_downloaded(pid))
check("missing raw slide does NOT produce an extract marker", not C.is_extracted(pid))
check("reported rather than silently skipped", "missing/corrupt" in out, out[-400:])

print("\n=== 4. one failing backbone does not poison the patient ===")
pid = "TCGA-ZZ-0005"
cases.append({"case_id": "c5", "submitter_id": pid})
C.MANIFEST.write_text(json.dumps(cases))
make_patient(pid, 1)
FAIL_BACKBONE["name"] = "convnext"
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    patched_main(["--backbones", ",".join(BB)])
out = buf.getvalue()
check("good backbone still saved",
      (C.FEATURES_PATCH_ROOT / "densenet121" / f"{pid}.npz").exists())
check("failing backbone saved nothing",
      not (C.FEATURES_PATCH_ROOT / "convnext" / f"{pid}.npz").exists())
check("patient NOT marked extracted while a backbone is missing", not C.is_extracted(pid))
check("slides KEPT so the failure can be retried",
      list((C.RAW_DIR / pid / "WSI").glob("*.svs")) != [], "slides were deleted!")

print("\n=== 5. retry fills in only the missing backbone ===")
FAIL_BACKBONE["name"] = None
before = (C.FEATURES_PATCH_ROOT / "densenet121" / f"{pid}.npz").stat().st_mtime_ns
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    patched_main(["--backbones", ",".join(BB)])
after = (C.FEATURES_PATCH_ROOT / "densenet121" / f"{pid}.npz").stat().st_mtime_ns
check("missing backbone now present",
      (C.FEATURES_PATCH_ROOT / "convnext" / f"{pid}.npz").exists())
check("already-good backbone was NOT recomputed", before == after)
check("patient now marked extracted", C.is_extracted(pid))
check("slides deleted now that everything succeeded",
      not list((C.RAW_DIR / pid / "WSI").glob("*.svs")))

print(f"\n{'='*60}")
print("FAILURES:", fails if fails else "none")
shutil.rmtree(root, ignore_errors=True)
sys.exit(1 if fails else 0)
