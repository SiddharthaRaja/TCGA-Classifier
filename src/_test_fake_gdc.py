#!/usr/bin/env python3
"""Fake GDC API so download_tcga.py can be exercised end to end offline."""
import json, threading, hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer

N_CASES = 6

CASES = [{"case_id": f"case-{i:03d}", "submitter_id": f"TCGA-ZZ-{i:04d}"}
         for i in range(N_CASES)]

def files_for(case_id, category, dtype, has_two_slides):
    idx = int(case_id.split("-")[1])
    if dtype == "Clinical Supplement":
        return [{"file_id": f"{case_id}-clin", "file_name": f"{case_id}.clin.xml", "file_size": 120}]
    if dtype == "Biospecimen Supplement":
        return [{"file_id": f"{case_id}-bio", "file_name": f"{case_id}.bio.xml", "file_size": 100}]
    if dtype == "Masked Copy Number Segment":
        if idx == 3:
            return []           # a case genuinely missing CNV
        return [{"file_id": f"{case_id}-cnv", "file_name": f"{case_id}.cnv.txt", "file_size": 200}]
    if dtype == "Slide Image":
        if idx == 4:
            return []           # a case with no diagnostic slide at all
        out = [{"file_id": f"{case_id}-wsi0", "file_name": f"{case_id}-0.svs", "file_size": 4096}]
        if has_two_slides and idx == 1:
            out.append({"file_id": f"{case_id}-wsi1", "file_name": f"{case_id}-1.svs", "file_size": 8192})
        return out
    return []

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        if self.path.endswith("/cases"):
            frm = payload.get("from", 0); size = payload.get("size", 500)
            hits = CASES[frm:frm + size]
            return self._json({"data": {"hits": hits, "pagination": {"total": len(CASES)}}})
        if self.path.endswith("/files"):
            conds = payload["filters"]["content"]
            g = {c["content"]["field"]: c["content"]["value"] for c in conds}
            hits = files_for(g["cases.case_id"], g["data_category"], g["data_type"],
                             self.server.two_slides)
            return self._json({"data": {"hits": hits}})
        self.send_error(404)

    def do_GET(self):
        fid = self.path.rsplit("/", 1)[-1]
        # deterministic bytes, correct size
        sizes = {"clin": 120, "bio": 100, "cnv": 200}
        if fid.endswith("-wsi0"): size = 4096
        elif fid.endswith("-wsi1"): size = 8192
        else: size = sizes[fid.rsplit("-", 1)[-1]]
        if fid == self.server.truncate_id:
            size = size // 2          # simulate a truncated transfer
        body = (hashlib.sha256(fid.encode()).digest() * (size // 32 + 1))[:size]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def serve(port=0, two_slides=True, truncate_id=None):
    srv = HTTPServer(("127.0.0.1", port), H)
    srv.two_slides = two_slides
    srv.truncate_id = truncate_id
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv

if __name__ == "__main__":
    import sys, time
    s = serve(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    print(s.server_address[1], flush=True)
    while True: time.sleep(1)
