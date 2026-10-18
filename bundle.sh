#!/bin/bash
S=$HOME/siddhartha/state
OUT=$HOME/siddhartha/backup_$(date +%Y%m%d_%H%M).tar.gz
cd "$S" && tar czf "$OUT" features_pooled patients thumbnails case_manifest.json 2>/dev/null
ls -lh "$OUT"
