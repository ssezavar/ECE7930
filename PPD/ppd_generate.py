#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PPD Synthetic Data Generator (Step-1 and Step-2)
# Usage: python ppd_generate.py --input dataset.csv --outdir artifacts --dry-run

import os, json, random, argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="PPD Synthetic Data Generator (EPDS-conditioned)")
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--outdir", default="artifacts", help="Output directory")
    ap.add_argument("--max-records", type=int, default=200, help="Process first N records")
    ap.add_argument("--dry-run", action="store_true", help="Use mock generator (no API calls)")
    ap.add_argument("--model", default="gpt-4o-mini", help="LLM model name if not dry-run")
    ap.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    return ap.parse_args()

def infer_category(text: str)->str:
    t = str(text).lower()
    if any(k in t for k in ["sleep","insomnia","tired","exhaust"]): return "Sleep"
    if any(k in t for k in ["anxiet","worry","panic"]): return "Anxiety"
    if any(k in t for k in ["cry","sad","miserable","hopeless","worthless","depress"]): return "Depression"
    if any(k in t for k in ["bond","attach","connect"]): return "Bonding"
    if any(k in t for k in ["support","alone","help","husband","partner","family"]): return "Support"
    if any(k in t for k in ["suicide","self-harm","kill myself","end my life"]): return "Suicidal Thoughts"
    return "General"

def infer_sentiment(text: str)->str:
    t = str(text).lower()
    if any(k in t for k in ["happy","hope","better","improve","relief"]): return "+1"
    if any(k in t for k in ["sad","cry","worthless","miserable","awful","hopeless","panic","anxiety","tired"]): return "-1"
    return "0"

def choose_epds_target(label: str, sentiment: str) -> int:
    if str(label).strip().lower() == "ppd":
        target = random.randint(13, 22)
    else:
        target = random.randint(3, 11)
    if sentiment == "-1":
        target = min(30, target + 2)
    elif sentiment == "+1":
        target = max(0, target - 2)
    return target

def ngram_overlap(a: str, b: str, n: int = 7) -> float:
    def grams(s):
        toks = s.split()
        return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))
    A = grams(a); B = grams(b)
    if not A or not B: return 0.0
    return len(A & B) / max(len(A), len(B))

def mock_step1_output(post: str, category: str, label: str, sentiment: str, score: int) -> Dict[str, Any]:
    synopsis_bits = []
    if category=="Sleep": synopsis_bits.append("sleep is fragmented")
    if category=="Depression": synopsis_bits.append("frequent crying and low mood")
    if category=="Anxiety": synopsis_bits.append("heightened worry around caregiving")
    if category=="Bonding": synopsis_bits.append("difficulty feeling close to the baby")
    if not synopsis_bits: synopsis_bits.append("ongoing fatigue and adjustment challenges")
    emotion = "hopeful" if sentiment == "+1" else ("flat" if sentiment == "0" else "distressed")
    sym = [category] if category in ["Sleep","Depression","Anxiety","Bonding","Support"] else ["Fatigue"]
    risk = []
    if any(k in post.lower() for k in ["alone","no help","unsupported"]): risk.append("Low_Support")
    red_flag = any(k in post.lower() for k in ["suicide","self-harm","kill myself","end my life"])
    return {
        "synopsis": f"A postpartum mother reports {', '.join(synopsis_bits)} while caring for her infant.",
        "emotion_summary": f"Tone appears {emotion}.",
        "symptom_tags": sym,
        "risk_factors": risk,
        "red_flag": bool(red_flag)
    }

def split_epds_items(total: int, red_flag: bool, tags: List[str]) -> List[int]:
    base = [0]*10
    focus = []
    if "Anxiety" in tags: focus += [3,4]
    if "Sleep" in tags: focus += [6]
    if "Depression" in tags or "Bonding" in tags: focus += [7,8]
    focus += [0,1,2,5]
    if red_flag: focus += [9]
    if not focus: focus = list(range(10))
    i=0
    while sum(base)<total and i<400:
        base[focus[i%len(focus)]] = min(3, base[focus[i%len(focus)]]+1)
        i+=1
    return base

def mock_step2_output(s1: Dict[str, Any], target_epds: int) -> Dict[str, Any]:
    core = "She reports"
    if "Anxiety" in s1.get("symptom_tags", []): core += " heightened worry during feeds"
    if "Sleep" in s1.get("symptom_tags", []): core += ", frequent night wakings"
    if "Depression" in s1.get("symptom_tags", []): core += ", and persistent low mood"
    if "Bonding" in s1.get("symptom_tags", []): core += ", with moments of feeling distant from the baby"
    weeks = random.choice([3,4,5,6,7,8])
    sleep_h = random.choice([2,3,4,5,6])
    feeding = random.choice(["Breastfeeding","Formula","Mixed"])
    support = random.choice(["Low","Medium","High"])
    marital = random.choice([True, False])
    items = split_epds_items(target_epds, s1.get("red_flag", False), s1.get("symptom_tags", []))
    return {
        "vignette": f"{weeks} weeks postpartum, the mother manages daily care with limited rest. {core}. Household routines feel heavy, and she postpones contact with friends. She tries brief rests between feeds but wakes feeling unrefreshed.",
        "postpartum_weeks": weeks,
        "sleep_hours": sleep_h,
        "feeding": feeding,
        "social_support": support,
        "marital_strain": marital,
        "symptom_tags": s1.get("symptom_tags", []),
        "risk_factors": s1.get("risk_factors", []),
        "epds_items": items,
        "epds_total": sum(items),
        "safety_flags": ["SelfHarm_Content"] if s1.get("red_flag", False) and sum(items) >= 19 else []
    }

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    if set(["Tweets","Labels"]).issubset(df.columns):
        df = df.rename(columns={"Tweets":"Post","Labels":"RawLabel"})
        df["Label"] = df["RawLabel"].apply(lambda x: "PPD" if str(x).strip().lower()=="postpartum" else "Non-PPD")
        df["Category"] = df["Post"].apply(infer_category)
        df["Sentiment"] = df["Post"].apply(infer_sentiment)
        df["Score"] = 0
    else:
        expected = ["Post","Category","Label","Sentiment","Score"]
        missing = [c for c in expected if c not in df.columns]
        if missing: raise ValueError(f"Missing expected columns: {missing}")

    if args.max_records and args.max_records>0:
        df = df.head(args.max_records).copy()

    f1 = open(outdir / "step1_synopses.jsonl", "w", encoding="utf-8")
    f2 = open(outdir / "step2_synthetic.jsonl", "w", encoding="utf-8")
    preview = []

    for idx, row in df.iterrows():
        s1 = mock_step1_output(row["Post"], row["Category"], row["Label"], row["Sentiment"], int(row["Score"]))
        target = choose_epds_target(row["Label"], row["Sentiment"])
        s2 = mock_step2_output(s1, target)
        overlap = ngram_overlap(str(row["Post"]), s2["vignette"], n=7)
        s2["_ngram_overlap_with_source"] = overlap
        f1.write(json.dumps({"id": int(idx), **s1}, ensure_ascii=False) + "\n")
        f2.write(json.dumps({"id": int(idx), **s2}, ensure_ascii=False) + "\n")
        preview.append({
            "id": int(idx),
            "label": row["Label"],
            "sentiment": row["Sentiment"],
            "post_excerpt": (row["Post"][:140]+"...") if len(row["Post"])>140 else row["Post"],
            "synopsis": s1["synopsis"],
            "vignette": s2["vignette"],
            "epds_total": s2["epds_total"],
            "ngram_overlap": overlap
        })

    f1.close(); f2.close()
    pd.DataFrame(preview).to_csv(outdir / "combined_preview.csv", index=False)
    print("Outputs written to", outdir)

if __name__ == "__main__":
    main()
