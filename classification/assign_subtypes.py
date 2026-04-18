#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

df = pd.read_csv("multimodal_dataset.csv")

def assign_subtype(row):
    er   = row.get("clin_er_status_binary", None)
    pr   = row.get("clin_pr_status_binary", None)
    her2 = row.get("clin_her2_status_binary", None)
    
    if er is None and pr is None and her2 is None:
        return None
    
    if her2 == 1 and er == 0 and pr == 0:
        return "HER2-enriched"
    elif er == 0 and pr == 0 and her2 == 0:
        return "Basal-like"
    elif (er == 1 or pr == 1) and her2 == 1:
        return "Luminal B"
    elif (er == 1 or pr == 1) and her2 == 0:
        return "Luminal A"
    else:
        return "Normal-like"

df["clin_pam50_subtype"] = df.apply(assign_subtype, axis=1)
print(df["clin_pam50_subtype"].value_counts())
df.to_csv("multimodal_dataset.csv", index=False)
print("Saved!")
