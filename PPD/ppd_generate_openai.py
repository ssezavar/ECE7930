#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PPD Synthetic Data Generator (OpenAI version): Step-1 (privacy synopsis) + Step-2 (EPDS-conditioned vignettes)
# Usage:
#   export OPENAI_API_KEY=YOUR_KEY
#   pip install --upgrade openai pandas
#   python ppd_generate_openai.py --input dataset.csv --outdir artifacts --model gpt-4o-mini --max-records 200

import os, re, json, time, argparse, random
from pathlib import Path
from typing import Dict, Any, List
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

STEP1_FEWSHOT = """### EXAMPLES ###
INPUT:
Post: I cry every night and can’t bond with my baby. My husband tries to help, but I still feel alone.
Category: Bonding
Label: PPD
Sentiment: -1
Score: 8
OUTPUT:
{"synopsis":"A few weeks after giving birth, the mother reports nightly crying and a persistent sense of disconnection from her baby despite spousal support.","emotion_summary":"Tone is sorrowful and lonely, showing guilt and helplessness.","symptom_tags":["Depression","Bonding"],"risk_factors":["Low_Support"],"red_flag":false}
### END EXAMPLES ###
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

def extract_json_line(text: str) -> str:
    # Try to find the last {...} JSON object
    mlist = re.findall(r'\{.*\}', text.strip(), flags=re.S)
    if not mlist:
        raise ValueError("No JSON object found in model output.")
    return mlist[-1]

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

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description="PPD Synthetic Data Generator (OpenAI)")
    ap.add_argument("--input", required=True, help="Path to CSV: (Tweets,Labels) or (Post,Category,Label,Sentiment,Score)")
    ap.add_argument("--outdir", default="artifacts", help="Output directory")
    ap.add_argument("--max-records", type=int, default=200, help="Process first N rows")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (e.g., gpt-4o, gpt-4o-mini)")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    # Accept two schemas
    if set(["Tweets","Labels"]).issubset(df.columns):
        df = df.rename(columns={"Tweets":"Post","Labels":"RawLabel"})
        df["Label"] = df["RawLabel"].apply(lambda x: "PPD" if str(x).strip().lower()=="postpartum" else "Non-PPD")
        df["Category"] = df["Post"].apply(infer_category)
        df["Sentiment"] = df["Post"].apply(infer_sentiment)
        df["Score"] = 0
    else:
        expected = ["Post","Category","Label","Sentiment","Score"]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

    if args.max_records and args.max_records > 0:
        df = df.head(args.max_records).copy()

    step1_file = open(outdir / "step1_synopses.jsonl", "w", encoding="utf-8")
    step2_file = open(outdir / "step2_synthetic.jsonl", "w", encoding="utf-8")

    preview_rows = []

    for idx, row in df.iterrows():
        # ---------------- Step 1 ----------------
        s1_user = (
            STEP1_TASK.replace("{POST_TEXT}", str(row["Post"]))
                      .replace("{CATEGORY}", str(row["Category"]))
                      .replace("{LABEL}", str(row["Label"]))
                      .replace("{SENTIMENT}", str(row["Sentiment"]))
                      .replace("{SCORE}", str(int(row["Score"])))
            + "\n" + STEP1_FEWSHOT + "\nOUTPUT:"
        )
        s1_json = call_openai_json(STEP1_SYSTEM, s1_user, args.model, args.temperature)

        # ---------------- Step 2 ----------------
        target_epds = choose_epds_target(row["Label"], row["Sentiment"])
        s2_user = (
            STEP2_TASK.replace("{SYNOPSIS_FROM_STEP1}", s1_json.get("synopsis",""))
                      .replace("{EMO_FROM_STEP1}", s1_json.get("emotion_summary",""))
                      .replace("{SYMPTOMS_FROM_STEP1}", json.dumps(s1_json.get("symptom_tags",[])))
                      .replace("{RISKS_FROM_STEP1}", json.dumps(s1_json.get("risk_factors",[])))
                      .replace("{TARGET_EPDS}", str(target_epds))
                      .replace("{WEEKS_OR_NULL}", "null")
                      .replace("{SLEEP_H_OR_NULL}", "null")
                      .replace("{FEEDING_OR_NULL}", "null")
                      .replace("{SUPPORT_OR_NULL}", "null")
                      .replace("{MARITAL_OR_NULL}", "null")
        )
        s2_json = call_openai_json(STEP2_SYSTEM, s2_user, args.model, args.temperature)

        # Ensure epds_total equals sum(items)
        if "epds_items" in s2_json and "epds_total" in s2_json:
            s = sum(int(x) for x in s2_json["epds_items"])
            if s2_json["epds_total"] != s:
                s2_json["epds_total"] = s

        # Simple privacy check: 7-gram overlap
        def ngram_overlap(a: str, b: str, n: int = 7) -> float:
            def grams(s):
                toks = s.split()
                return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))
            A = grams(a); B = grams(b)
            if not A or not B: return 0.0
            return len(A & B) / max(len(A), len(B))
        s2_json["_ngram_overlap_with_source"] = ngram_overlap(str(row["Post"]), s2_json.get("vignette",""), 7)

        # Write JSONL
        step1_file.write(json.dumps({"id": int(idx), **s1_json}, ensure_ascii=False) + "\n")
        step2_file.write(json.dumps({"id": int(idx), **s2_json}, ensure_ascii=False) + "\n")

        # Preview
        preview_rows.append({
            "id": int(idx),
            "label": row["Label"],
            "sentiment": row["Sentiment"],
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
