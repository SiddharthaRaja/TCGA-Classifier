#!/usr/bin/env python3
import requests
import pandas as pd
from pathlib import Path

patients = ['TCGA-A2-A0CV', 'TCGA-A2-A1G4', 'TCGA-A2-A4S2', 'TCGA-A7-A0DC', 'TCGA-A8-A08R', 'TCGA-A8-A09R', 'TCGA-AN-A0FY', 'TCGA-AO-A0JF', 'TCGA-AO-A12H', 'TCGA-AO-A1KR', 'TCGA-AQ-A1H3', 'TCGA-AR-A24H', 'TCGA-AR-A24U', 'TCGA-B6-A0IM', 'TCGA-B6-A40C', 'TCGA-BH-A0DV', 'TCGA-BH-A0E6', 'TCGA-BH-A1F2', 'TCGA-BH-A204', 'TCGA-C8-A3M8', 'TCGA-D8-A143', 'TCGA-D8-A1XC', 'TCGA-D8-A1XY', 'TCGA-D8-A27K', 'TCGA-D8-A27T', 'TCGA-E2-A15D', 'TCGA-E2-A15K', 'TCGA-E2-A1IH', 'TCGA-E2-A1LH', 'TCGA-E2-A1LK', 'TCGA-E9-A247', 'TCGA-E9-A3Q9', 'TCGA-GM-A2DM', 'TCGA-LL-A440', 'TCGA-LL-A442', 'TCGA-OL-A5S0', 'TCGA-OL-A66P', 'TCGA-XX-A899']

payload = {
    "filters": {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.submitter_id", "value": patients}},
            {"op": "=", "content": {"field": "project.project_id", "value": "TCGA-BRCA"}}
        ]
    },
    "fields": "submitter_id,diagnoses.subtype,demographic.gender",
    "expand": "diagnoses",
    "size": 100,
    "format": "json"
}

r = requests.post("https://api.gdc.cancer.gov/cases", json=payload, timeout=60)
data = r.json()

records = []
for hit in data["data"]["hits"]:
    pid = hit["submitter_id"]
    subtype = None
    for diag in hit.get("diagnoses", []):
        subtype = diag.get("subtype") or diag.get("molecular_subtype")
        if subtype:
            break
    records.append({"patient_id": pid, "pam50_subtype": subtype})

df = pd.DataFrame(records)
print(df)
df.to_csv("pam50_labels.csv", index=False)
print(f"\nSaved pam50_labels.csv")
