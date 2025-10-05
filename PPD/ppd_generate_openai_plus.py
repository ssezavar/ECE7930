#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PPD Synthetic Data Generator (OpenAI+) — supports CSV or JSONL input, and per-row controls.
# Usage examples:
#   export OPENAI_API_KEY=YOUR_KEY
#   pip install --upgrade openai pandas
#   python ppd_generate_openai_plus.py --input dataset.csv --outdir artifacts --model gpt-4o-mini
#   python ppd_generate_openai_plus.py --jsonl-input posts.jsonl --outdir artifacts --model gpt-4o-mini
#   python ppd_generate_openai_plus.py --input dataset.csv --controls controls.csv --outdir artifacts --model gpt-4o-mini
#
# Controls CSV columns (optional):
#   row_id,target_epds,postpartum_weeks,sleep_hours,feeding,social_support,marital_strain
#   - row_id must match the zero-based row index of the input after loading (head() and filtering apply!).
#   - Any column may be left blank to keep default behavior.

import os, re, json, time, argparse, random
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# --------------------- Prompt blocks ---------------------

STEP1_SYSTEM = "You generate privacy-preserving postpartum vignettes for research. Only output the requested JSON."
STEP1_TASK = """INPUT FIELDS
Post:{POST_TEXT}
Category:{CATEGORY}
Label:{LABEL}
Sentiment:{SENTIMENT}
Score:{SCORE}

TASK
1) Write a concise third-person synopsis of the mother's situation (2–4 sentences). Keep postpartum context central (sleep/feeding context if implied, social/marital support if implied). Remove any names, locations, ages, or uniquely identifying details.
2) Summarize the emotional tone and salient emotions in 1–2 sentences (no diagnosis).
3) Extract normalized tags from the input:
   - symptom_tags: subset of {Depression, Anxiety, Fatigue, Bonding, Sleep, Guilt, Irritability, Appetite, Concentration, Suicidal_Ideation, Support}
   - risk_factors: any of {Low_Support, Marital_Strain, Sleep_Fragmentation, NICU_Stress, Breastfeeding_Difficulty, Birth_Complications, Financial_Stress}
4) Red-flag detection: suicidal ideation/self-harm present? return true/false.

OUTPUT FORMAT (STRICT, ONE LINE, NO WHITESPACE)
{"synopsis":"...","emotion_summary":"...","symptom_tags":[...],"risk_factors":[...],"red_flag":false}
"""

STEP1_FEWSHOT_MINI = """### EXAMPLE ###
INPUT:
Post: I cry every night and can’t bond with my baby. I still feel alone.
Category: Bonding
Label: PPD
Sentiment: -1
Score: 8
OUTPUT:
{"synopsis":"A few weeks after giving birth, the mother reports nightly crying and a sense of disconnection from her baby despite attempts to cope.","emotion_summary":"Tone is sorrowful and lonely.","symptom_tags":["Depression","Bonding"],"risk_factors":[],"red_flag":false}
### END EXAMPLE ###
"""

STEP2_SYSTEM = "You generate synthetic postpartum vignettes for research. Do not copy the input; output only the requested JSON."
STEP2_TASK = """INPUT
Original_Synopsis:{SYNOPSIS_FROM_STEP1}
Emotion_Summary:{EMO_FROM_STEP1}
Symptom_Tags:{SYMPTOMS_FROM_STEP1}
Risk_Factors:{RISKS_FROM_STEP1}
Target_EPDS_Total:{TARGET_EPDS}
Optional_Controls:
- postpartum_weeks:{WEEKS_OR_NULL}
- sleep_hours:{SLEEP_H_OR_NULL}
- feeding:{FEEDING_OR_NULL}
- social_support:{SUPPORT_OR_NULL}
- marital_strain:{MARITAL_OR_NULL}

RULES
A) Create a fresh synthetic vignette (4–6 sentences) centered on postpartum context; match severity to Target_EPDS_Total; be coherent with Symptom_Tags/Risk_Factors but change particulars.
B) Output EPDS item scores (10 items) that sum exactly to Target_EPDS_Total; map content to items (anxiety→4–5, sleep→7, sadness/crying→8–9, self-harm only if justified).
C) Safety: If self-harm content would be implied, keep language clinically neutral and set "safety_flags":["SelfHarm_Content"].
D) No PII, no dates/locations, no diagnosis language.

OUTPUT (STRICT, ONE LINE, NO WHITESPACE)
{"vignette":"...","postpartum_weeks":X,"sleep_hours":X,"feeding":"...","social_support":"...","marital_strain":false,"symptom_tags":[...],"risk_factors":[...],"epds_items":[i1,i2,i3,i4,i5,i6,i7,i8,i9,i10],"epds_total":N,"safety_flags":[...]}
"""

# --------------------- Helpers ---------------------

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

def guess_target_epds_from_sentiment(sentiment: str)->int:
    if sentiment == "-1":
        return random.randint(12, 20)
    if sentiment == "+1":
        return random.randint(2, 8)
    return random.randint(6, 12)

def choose_epds_target(label: Optional[str], sentiment: str) -> int:
    if label and str(label).strip().lower() == "ppd":
        target = random.randint(13, 22)
    elif label and str(label).strip().lower() == "non-ppd":
        target = random.randint(3, 11)
    else:
        target = guess_target_epds_from_sentiment(sentiment)
    return target

def call_openai_json(system: str, user: str, model: str, temperature: float, max_retries: int = 3) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content":user}
                ]
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("OpenAI call failed after retries.")

def load_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(["Tweets","Labels"]).issubset(df.columns):
        df = df.rename(columns={"Tweets":"Post","Labels":"RawLabel"})
        df["Label"] = df["RawLabel"].apply(lambda x: "PPD" if str(x).strip().lower()=="postpartum" else "Non-PPD")
        df["Category"] = df["Post"].apply(infer_category)
        df["Sentiment"] = df["Post"].apply(infer_sentiment)
        df["Score"] = 0
    else:
        # Require the richer schema
        expected = ["Post","Category","Label","Sentiment","Score"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing expected columns: {missing}")
    return df

def load_from_jsonl(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    # Accept 'Post' or 'text'
    if "Post" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text":"Post"})
    if "Post" not in df.columns:
        raise ValueError("JSONL must contain 'Post' or 'text' field.")
    # Optional label fields
    if "Label" not in df.columns:
        if "label" in df.columns:
            df["Label"] = df["label"].apply(lambda x: "PPD" if str(x).strip().lower()=="postpartum" else "Non-PPD")
        elif "raw_label" in df.columns:
            df["Label"] = df["raw_label"].apply(lambda x: "PPD" if str(x).strip().lower()=="postpartum" else "Non-PPD")
        else:
            df["Label"] = "Unknown"
    if "Category" not in df.columns:
        df["Category"] = df["Post"].apply(infer_category)
    if "Sentiment" not in df.columns:
        df["Sentiment"] = df["Post"].apply(infer_sentiment)
    if "Score" not in df.columns:
        df["Score"] = 0
    return df

def load_controls(path: Path) -> Dict[int, Dict[str, Any]]:
    cdf = pd.read_csv(path)
    key_col = "row_id" if "row_id" in cdf.columns else ("id" if "id" in cdf.columns else ("index" if "index" in cdf.columns else None))
    if key_col is None:
        raise ValueError("Controls CSV must include a row_id (or id/index) column.")
    cdf = cdf.set_index(key_col)
    controls = {}
    for ridx, row in cdf.iterrows():
        d = {}
        for k in ["target_epds","postpartum_weeks","sleep_hours","feeding","social_support","marital_strain"]:
            if k in row and pd.notnull(row[k]):
                d[k] = row[k]
        controls[int(ridx)] = d
    return controls

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description="PPD Synthetic Data Generator (OpenAI+, CSV/JSONL + Controls)")
    ap.add_argument("--input", help="CSV path (Tweets,Labels) or (Post,Category,Label,Sentiment,Score)")
    ap.add_argument("--jsonl-input", help="JSONL path with Post/text [+ optional label]")
    ap.add_argument("--outdir", default="artifacts", help="Output directory")
    ap.add_argument("--controls", help="Optional controls CSV (row_id,...). See header in file.")
    ap.add_argument("--max-records", type=int, default=200, help="Process first N rows")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    args = ap.parse_args()

    if not args.input and not args.jsonl_input:
        raise SystemExit("Provide --input CSV or --jsonl-input JSONL")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.input:
        df = load_from_csv(Path(args.input))
    else:
        df = load_from_jsonl(Path(args.jsonl_input))

    if args.max_records and args.max_records > 0:
        df = df.head(args.max_records).copy()

    # Controls (optional)
    ctrl = {}
    if args.controls:
        ctrl = load_controls(Path(args.controls))

    step1_file = open(outdir / "step1_synopses.jsonl", "w", encoding="utf-8")
    step2_file = open(outdir / "step2_synthetic.jsonl", "w", encoding="utf-8")

    preview_rows = []

    for ridx, row in df.reset_index(drop=True).iterrows():
        # ---------------- Step 1 ----------------
        s1_user = (
            STEP1_TASK.replace("{POST_TEXT}", str(row["Post"]))
                      .replace("{CATEGORY}", str(row["Category"]))
                      .replace("{LABEL}", str(row["Label"]))
                      .replace("{SENTIMENT}", str(row["Sentiment"]))
                      .replace("{SCORE}", str(int(row["Score"])))
            + "\n" + STEP1_FEWSHOT_MINI + "\nOUTPUT:"
        )
        s1_json = call_openai_json(STEP1_SYSTEM, s1_user, args.model, args.temperature)

        # ---------------- Step 2 ----------------
        overrides = ctrl.get(int(ridx), {})
        target_epds = int(overrides.get("target_epds", choose_epds_target(row.get("Label","Unknown"), row.get("Sentiment","0"))))

        def fmt(v):
            if v is None or (isinstance(v, float) and pd.isna(v)): return "null"
            if isinstance(v, (int, float)): return str(int(v))
            if isinstance(v, str):
                if v.lower() in ["true", "false"]: return v.lower()
                return v
            return "null"

        weeks_val  = fmt(overrides.get("postpartum_weeks", "null"))
        sleep_val  = fmt(overrides.get("sleep_hours", "null"))
        feed_val   = overrides.get("feeding", "null")
        support_val= overrides.get("social_support", "null")
        marital_val= str(overrides.get("marital_strain", "null")).lower()

        s2_user = (
            STEP2_TASK.replace("{SYNOPSIS_FROM_STEP1}", s1_json.get("synopsis",""))
                      .replace("{EMO_FROM_STEP1}", s1_json.get("emotion_summary",""))
                      .replace("{SYMPTOMS_FROM_STEP1}", json.dumps(s1_json.get("symptom_tags",[])))
                      .replace("{RISKS_FROM_STEP1}", json.dumps(s1_json.get("risk_factors",[])))
                      .replace("{TARGET_EPDS}", str(target_epds))
                      .replace("{WEEKS_OR_NULL}", weeks_val)
                      .replace("{SLEEP_H_OR_NULL}", sleep_val)
                      .replace("{FEEDING_OR_NULL}", feed_val)
                      .replace("{SUPPORT_OR_NULL}", support_val)
                      .replace("{MARITAL_OR_NULL}", marital_val)
        )
        s2_json = call_openai_json(STEP2_SYSTEM, s2_user, args.model, args.temperature)

        # Ensure epds_total equals sum(items)
        if "epds_items" in s2_json and "epds_total" in s2_json:
            s = sum(int(x) for x in s2_json["epds_items"])
            if s2_json["epds_total"] != s:
                s2_json["epds_total"] = s

        # Privacy check (7-gram overlap)
        def ngram_overlap(a: str, b: str, n: int = 7) -> float:
            def grams(s):
                toks = s.split()
                return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))
            A = grams(a); B = grams(b)
            if not A or not B: return 0.0
            return len(A & B) / max(len(A), len(B))
        s2_json["_ngram_overlap_with_source"] = ngram_overlap(str(row["Post"]), s2_json.get("vignette",""), 7)

        # Write JSONL
        step1_file.write(json.dumps({"row_id": int(ridx), **s1_json}, ensure_ascii=False) + "\n")
        step2_file.write(json.dumps({"row_id": int(ridx), **s2_json}, ensure_ascii=False) + "\n")

        # Preview
        preview_rows.append({
            "row_id": int(ridx),
            "label": row.get("Label"),
            "sentiment": row.get("Sentiment"),
            "post_excerpt": (row["Post"][:140] + "...") if len(row["Post"])>140 else row["Post"],
            "synopsis": s1_json.get("synopsis",""),
            "vignette": s2_json.get("vignette",""),
            "epds_total": s2_json.get("epds_total", None),
            "ngram_overlap": s2_json.get("_ngram_overlap_with_source", 0.0)
        })

    step1_file.close()
    step2_file.close()

    pd.DataFrame(preview_rows).to_csv(outdir / "combined_preview.csv", index=False)
    print("Done. Files written to", outdir)

if __name__ == "__main__":
    main()
