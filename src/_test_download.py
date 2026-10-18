#!/usr/bin/env python3
"""End-to-end test of download_tcga.py against the fake GDC, covering the
failure modes that actually bit this project."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import _test_fake_gdc as fake

root = Path(tempfile.mkdtemp(prefix="tcga_t_"))
DATA = root / "data"; STATE = root / "state"
env = dict(os.environ, TCGA_DATA_ROOT=str(DATA), TCGA_STATE_ROOT=str(STATE),
           GDC_MAX_RETRIES="2", GDC_BACKOFF="0")

srv = fake.serve()
port = srv.server_address[1]
env["GDC_API"] = f"http://127.0.0.1:{port}"

def run(script, *args, quiet=True):
    r = subprocess.run([sys.executable, str(HERE / script), *args],
                       env=env, capture_output=True, text=True, cwd=HERE)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"{script} {args} exited {r.returncode}")
    return r.stdout

fails = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail and not cond else ''}")
    if not cond: fails.append(name)

print("\n=== 1. manifest + first full download ===")
run("download_tcga.py", "--manifest")
manifest = json.loads((STATE / "case_manifest.json").read_text())
check("manifest built with 6 cases", len(manifest) == 6)

out = run("download_tcga.py")
markers = sorted(p.parent.name for p in STATE.glob("patients/*/download_complete.json"))
check("all 6 patients marked downloaded", len(markers) == 6, str(markers))
check("reports 6 new", "new: 6" in out, out[-400:])

m1 = json.loads((STATE / "patients/TCGA-ZZ-0001/download_complete.json").read_text())
check("two-slide patient records both slides", len(m1["files"]["wsi"]) == 2,
      str(m1["files"]["wsi"]))
m4 = json.loads((STATE / "patients/TCGA-ZZ-0004/download_complete.json").read_text())
check("no-slide patient records empty wsi list", m4["files"]["wsi"] == [])
m3 = json.loads((STATE / "patients/TCGA-ZZ-0003/download_complete.json").read_text())
check("missing-CNV patient still completes", m3["modality_stats"]["cnv"] == "none")

check("small artifacts mirrored into state",
      (STATE / "patients/TCGA-ZZ-0000/Clinical").exists()
      and (STATE / "patients/TCGA-ZZ-0000/Copy_Number_Variation").exists())

print("\n=== 2. rerun is a no-op (no repeated work) ===")
out = run("download_tcga.py")
check("second run downloads nothing new", "new: 0" in out and "already done: 6" in out,
      out[-300:])

print("\n=== 3. extracted patient is never re-downloaded ===")
pid = "TCGA-ZZ-0000"
(STATE / "patients" / pid / "extract_complete.json").write_text('{"patient_id":"%s"}' % pid)
for f in (DATA / "raw" / pid / "WSI").glob("*.svs"):
    f.unlink()                                  # exactly what extraction does
out = run("download_tcga.py")
still_gone = not any((DATA / "raw" / pid / "WSI").glob("*.svs"))
check("deleted slide of an extracted patient is NOT re-downloaded", still_gone,
      "slide came back -> infinite re-download loop")

print("\n=== 4. interrupted download leaves no marker and is retried ===")
pid = "TCGA-ZZ-0002"
shutil.rmtree(STATE / "patients" / pid)
shutil.rmtree(DATA / "raw" / pid)
srv.truncate_id = f"case-002-wsi0"              # server returns half a file
out = run("download_tcga.py")
check("truncated download writes NO marker",
      not (STATE / "patients" / pid / "download_complete.json").exists())
check("truncated download counted as failed", "failed (will retry): 1" in out, out[-300:])
srv.truncate_id = None
out = run("download_tcga.py")
check("retry after truncation succeeds",
      (STATE / "patients" / pid / "download_complete.json").exists())
m = json.loads((STATE / "patients" / pid / "download_complete.json").read_text())
sz = (DATA / "raw" / pid / "WSI" / "case-002-0.svs").stat().st_size
check("retried file is the full size", sz == 4096, f"got {sz}")

print("\n=== 5. lock prevents double work ===")
(STATE / "locks").mkdir(parents=True, exist_ok=True)
pid = "TCGA-ZZ-0005"
shutil.rmtree(STATE / "patients" / pid)
(STATE / "locks" / f"{pid}.lock").write_text('{"job":"other"}')
out = run("download_tcga.py")
check("locked patient is skipped, not processed", "locked by another job" in out,
      out[-400:])
(STATE / "locks" / f"{pid}.lock").unlink()

print("\n=== 6. --status reflects markers, not directory contents ===")
out = run("download_tcga.py", "--status")
print("     " + "\n     ".join(l for l in out.splitlines() if ":" in l and "root" not in l.lower()))
check("status counts the extracted patient as extracted", "features extracted       : 1" in out)

print("\n=== 7. --limit ===")
shutil.rmtree(STATE / "patients" / "TCGA-ZZ-0003", ignore_errors=True)
shutil.rmtree(STATE / "patients" / "TCGA-ZZ-0005", ignore_errors=True)
out = run("download_tcga.py", "--limit", "1")
check("--limit 1 stops after one new patient", "new: 1" in out, out[-300:])

print(f"\n{'='*60}")
print("FAILURES:", fails if fails else "none")
shutil.rmtree(root, ignore_errors=True)
sys.exit(1 if fails else 0)
