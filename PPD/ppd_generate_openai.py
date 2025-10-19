#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppd_generate.py

Outputs (in --outdir, default: artifacts/):
  - step1_synopses.jsonl
  - step2_synthetic.jsonl
  - combined_preview.csv

Usage:
  python ppd_generate.py --input dataset.csv --outdir artifacts --dry-run
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Tuple, Optional

# OpenAI client 
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_AVAILABLE = False

# ------------------------------- CLI & Config ----------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPD synthetic data generator")
    p.add_argument("--input", required=True, help="Path to input CSV (expects columns: Post, Label, Category, Sentiment)")
    p.add_argument("--outdir", default="artifacts", help="Directory for outputs (default: artifacts)")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model to use (default: gpt-4o-mini)")
    p.add_argument("--dry-run", action="store_true", help="Use mock outputs instead of calling the API")
    p.add_argument("--max-rows", type=int, default=None, help="Limit rows for quick tests")
    p.add_argument("--dedup-threshold", type=float, default=0.90, help="Jaccard overlap threshold to flag duplicates (default: 0.90)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for mock generation")
    p.add_argument("--column-map", default=None, help="Optional mapping like: post=Text,label=Target,category=Topic,sentiment=Polarity")
    return p.parse_args()

# ------------------------------- Utilities -------------------------------------

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _resolve_columns(fieldnames, user_map: Optional[Dict[str,str]]=None):
    # normalize: lowercase & strip spaces
    fnorm = { (f or "").strip().lower(): f for f in (fieldnames or []) }

    # allow explicit mapping via --column-map like: post=Text,label=Target,category=Topic,sentiment=Polarity
    resolved = {}
    if user_map:
        for need, have in user_map.items():
            if have in fieldnames:
                resolved[need] = have

    # common synonyms (lowercased keys of CSV)
    synonyms = {
        "post": ["post", "text", "content", "post_text", "message", "body"],
        "label": ["label", "class", "target", "ppd_label", "y"],
        "category": ["category", "topic", "tag", "domain"],
        "sentiment": ["sentiment", "polarity", "sent", "emotion"],
    }

    for need, cands in synonyms.items():
        if need in resolved:
            continue
        for c in cands:
            if c in fnorm:
                resolved[need] = fnorm[c]
                break

    missing = [k for k in ["post","label","category","sentiment"] if k not in resolved]
    return resolved, missing


def read_rows(csv_path: str, max_rows: Optional[int] = None, column_map: Optional[str]=None) -> List[Dict[str, str]]:
    # parse user mapping
    user_map = None
    if column_map:
        user_map = {}
        for kv in column_map.split(","):
            if "=" in kv:
                k,v = kv.split("=",1)
                user_map[k.strip().lower()] = v.strip()

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        
        # Sara: resolve columns flexibly
        resolved, missing = _resolve_columns(r.fieldnames, user_map=user_map)
        if missing:
            raise ValueError(
                "Input CSV missing required columns. "
                f"Need logical columns: ['Post','Label','Category','Sentiment']\n"
                f"Your header: {r.fieldnames}\n"
                "Tip: rename headers OR pass --column-map post=YourPost,label=YourLabel,category=YourCategory,sentiment=YourSentiment"
            )
            
        # Sara: read with resolved names 10/13/25
        for i, row in enumerate(r):
            rows.append({
                "Post": (row.get(resolved["post"]) or "").strip(),
                "Label": (row.get(resolved["label"]) or "").strip(),
                "Category": (row.get(resolved["category"]) or "").strip(),
                "Sentiment": (row.get(resolved["sentiment"]) or "").strip(),
            })
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows

def write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def normalize_text(s: str) -> List[str]:
    # Lowercase, remove punctuation, split into tokens
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = re.split(r"\s+", s)
    return [t for t in tokens if t]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def is_duplicate(candidate: str, seen_tokens: List[List[str]], threshold: float) -> bool:
    cand_tokens = normalize_text(candidate)
    for toks in seen_tokens:
        if jaccard(cand_tokens, toks) >= threshold:
            return True
    return False

def choose_epds_target(label: str, sentiment: str) -> str:
    """
    Simple mapping to a target EPDS-like bucket or guidance tag.
    Customize as needed.
    """
    L = (label or "").strip().lower()
    S = (sentiment or "").strip().lower()

    if L in {"ppd", "positive_ppd", "depressed", "high_risk"}:
        return "EPDS>=13 (probable depression) – urgent support focus"
    if L in {"possible_ppd", "moderate_risk"} or S in {"negative"}:
        return "EPDS 10–12 (possible depression) – screening + resources"
    if S in {"neutral"}:
        return "EPDS 5–9 (watchful waiting) – psychoeducation"
    return "EPDS 0–4 (reassure) – general wellness"

# ------------------------------- Mock Generators --------------------------------
# 10/9/25
MOCK_FEELINGS = [
    "I feel overwhelmed and exhausted after sleepless nights.",
    "I’m anxious that I’m not bonding well with my baby.",
    "Feeding times are stressful and I doubt myself a lot.",
    "I’m irritable with everyone and then feel guilty.",
    "I keep crying without knowing exactly why.",
]

MOCK_CONTEXT = [
    "partner works late shifts", "limited family support", "cesarean recovery pain",
    "breastfeeding latch difficulties", "baby colic", "language barrier in clinic"
]

def mock_step1_output(post: str, category: str, label: str, sentiment: str) -> Dict:
    syn = {
        "type": "synopsis",
        "summary": f"Parent reports: {random.choice(MOCK_FEELINGS)}",
        "salient_factors": random.sample(MOCK_CONTEXT, k=min(2, len(MOCK_CONTEXT))),
        "estimated_risk": choose_epds_target(label, sentiment),
        "source_hash": hashlib.md5(post.encode("utf-8")).hexdigest(),
    }
    return syn

def mock_step2_output(s1: Dict, target: str) -> Dict:
    body = (
        f"{s1.get('summary','')} Context: {', '.join(s1.get('salient_factors', []))}. "
        f"Guidance: {target}. Example self-talk: "
        f"“I’m doing my best; it’s okay to ask for help. I’ll call the nurse line if I feel worse.”"
    )
    return {
        "type": "synthetic_post",
        "synthetic_text": body,
        "target": target,
        "style": "first-person postpartum diary",
    }

# ------------------------------- OpenAI Calls -----------------------------------
# adding the openai call  10/10/25
def make_openai_client():
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI library not available in this environment.")
    return OpenAI()

def step1_with_llm(client, model: str, post: str, category: str, label: str, sentiment: str) -> Dict:
    """
    Step 1: Turn the raw post into a concise clinical synopsis + salient factors (no PHI).
    """
    prompt = f"""
You are assisting with postpartum mental health data abstraction (no PHI).
Summarize the user's post into a brief, clinician-friendly synopsis and list 2–4 salient contextual factors.
Then estimate a coarse EPDS risk bucket label (do NOT output a number), using categories like:
- "EPDS>=13 (probable depression) – urgent support focus"
- "EPDS 10–12 (possible depression) – screening + resources"
- "EPDS 5–9 (watchful waiting) – psychoeducation"
- "EPDS 0–4 (reassure) – general wellness"

INPUT FIELDS:
- Post: {post}
- Category: {category}
- Provided Label: {label}
- Sentiment: {sentiment}

Return strict JSON with keys: summary (string), salient_factors (array of strings), estimated_risk (string).
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()
    # Try to parse JSON from the response. If it isn't JSON, wrap it.
    try:
        parsed = json.loads(content)
        summary = (parsed.get("summary") or "").strip()
        salient = parsed.get("salient_factors") or []
        risk = (parsed.get("estimated_risk") or "").strip()
    except Exception:
        # Fallback: build a light JSON wrapper if the model returned text.
        summary = content[:500]
        salient = []
        risk = choose_epds_target(label, sentiment)

    return {
        "type": "synopsis",
        "summary": summary,
        "salient_factors": salient,
        "estimated_risk": risk,
        "source_hash": hashlib.md5(post.encode("utf-8")).hexdigest(),
    }

def step2_with_llm(client, model: str, s1: Dict, target: str) -> Dict:
    """
    Step 2: Generate a synthetic first-person postpartum diary-style sample aligned with target.
    """
    prompt = f"""
You will write a short, empathetic, first-person postpartum diary entry (100–180 words).
Use the synopsis and salient factors below. Avoid PHI or realistic identifiers.
The tone should be human, not clinical, and align with the guidance target.

SYNOPSIS: {json.dumps(s1, ensure_ascii=False)}
TARGET: {target}

Return ONLY the diary text, no preface or JSON.
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    text = resp.choices[0].message.content.strip()
    return {
        "type": "synthetic_post",
        "synthetic_text": text,
        "target": target,
        "style": "first-person postpartum diary",
    }

# ------------------------------- Main Pipeline ----------------------------------
#modified for dataset flexibility, Sara 10/14/25
def main():
    args = parse_args()
    random.seed(args.seed)
    ensure_outdir(args.outdir)

    print("=" * 70)
    if args.dry_run:
        print("[mode] DRY-RUN mode activated → using mock synthetic data (no OpenAI API).")
        print("         This uses a simple rule-based algorithm for generating outputs.")
    else:
        print(f"[mode] LIVE mode → using real OpenAI model: {args.model}")
    print("=" * 70)

    # Output paths
    path_s1 = os.path.join(args.outdir, "step1_synopses.jsonl")
    path_s2 = os.path.join(args.outdir, "step2_synthetic.jsonl")
    path_combo = os.path.join(args.outdir, "combined_preview.csv")

    #rows = read_rows(args.input, max_rows=args.max_rows)
    # Sara 10/14/2025
    rows = read_rows(args.input, max_rows=args.max_rows, column_map=args.column_map)


    # Prepare OpenAI client if needed
    client = None
    if not args.dry_run:
        try:
            client = make_openai_client()
        except Exception as e:
            print(f"[warn] Could not initialize OpenAI client ({e}). Switching to --dry-run mode.")
            args.dry_run = True
            print("[mode] Fallback → DRY-RUN mode (mock data generation).")

    step1_records: List[Dict] = []
    step2_records: List[Dict] = []
    combined_rows: List[Dict[str, str]] = []

    # Simple dedup via Jaccard over normalized tokens
    seen_s1_tokens: List[List[str]] = []
    seen_s2_tokens: List[List[str]] = []

    for idx, row in enumerate(rows):
        post = row["Post"]
        category = row["Category"]
        label = row["Label"]
        sentiment = row["Sentiment"]

        s1 = None
        s2 = None
        used_mock = False
        skip_reason = ""

        try:
            # ---- STEP 1 ----
            if args.dry_run:
                s1 = mock_step1_output(post, category, label, sentiment)
                used_mock = True
            else:
                try:
                    s1 = step1_with_llm(client, args.model, post, category, label, sentiment)
                except Exception as e1:
                    emsg = str(e1).lower()
                    if ("insufficient_quota" in emsg) or ("invalid_api_key" in emsg):
                        print(f"[warn] step1 idx={idx}: API error '{e1}'. Using mock output for this row.")
                        s1 = mock_step1_output(post, category, label, sentiment)
                        used_mock = True
                    else:
                        # Non-quota errors still shouldn't crash the whole run; attempt mock
                        print(f"[warn] step1 idx={idx}: unexpected error '{e1}'. Using mock.")
                        s1 = mock_step1_output(post, category, label, sentiment)
                        used_mock = True

            # Dedup check for STEP 1
            if is_duplicate(s1.get("summary", ""), seen_s1_tokens, args.dedup_threshold):
                skip_reason = "dup_step1"
            else:
                seen_s1_tokens.append(normalize_text(s1.get("summary", "")))
                step1_records.append(s1)

            # ---- STEP 2 ----
            target = s1.get("estimated_risk") or choose_epds_target(label, sentiment)
            if args.dry_run:
                s2 = mock_step2_output(s1, target)
                used_mock = True
            else:
                try:
                    s2 = step2_with_llm(client, args.model, s1, target)
                except Exception as e2:
                    emsg = str(e2).lower()
                    if ("insufficient_quota" in emsg) or ("invalid_api_key" in emsg):
                        print(f"[warn] step2 idx={idx}: API error '{e2}'. Using mock output for this row.")
                        s2 = mock_step2_output(s1, target)
                        used_mock = True
                    else:
                        print(f"[warn] step2 idx={idx}: unexpected error '{e2}'. Using mock.")
                        s2 = mock_step2_output(s1, target)
                        used_mock = True

            # Dedup check for STEP 2
            if is_duplicate(s2.get("synthetic_text", ""), seen_s2_tokens, args.dedup_threshold):
                skip_reason = "dup_step2" if not skip_reason else f"{skip_reason}|dup_step2"
            else:
                seen_s2_tokens.append(normalize_text(s2.get("synthetic_text", "")))
                step2_records.append(s2)

        except Exception as fatal:
            # Extremely defensive: never crash the whole job
            print(f"[error] idx={idx}: fatal loop error '{fatal}'. Skipping row.")
            skip_reason = (skip_reason + "|fatal").strip("|")

        # ---- Combined preview row ----
        combined_rows.append({
            "id": str(idx),
            "used_mock": "yes" if used_mock else "no",
            "skip_reason": skip_reason,
            "label": label,
            "sentiment": sentiment,
            "category": category,
            "post": post,
            "s1_summary": (s1 or {}).get("summary", ""),
            "s1_salient": "; ".join((s1 or {}).get("salient_factors", [])),
            "s1_estimated_risk": (s1 or {}).get("estimated_risk", ""),
            "s2_text": (s2 or {}).get("synthetic_text", ""),
            "s2_target": (s2 or {}).get("target", ""),
            "s2_style": (s2 or {}).get("style", ""),
        })

        if idx % 25 == 0:
            print(f"[info] processed {idx+1}/{len(rows)} rows")

    # ---------------- Write Artifacts ----------------
    write_jsonl(path_s1, step1_records)
    write_jsonl(path_s2, step2_records)

    combo_fields = [
        "id", "used_mock", "skip_reason",
        "label", "sentiment", "category", "post",
        "s1_summary", "s1_salient", "s1_estimated_risk",
        "s2_text", "s2_target", "s2_style",
    ]
    write_csv(path_combo, combined_rows, combo_fields)

    print(f"[done] Wrote:\n  - {path_s1}\n  - {path_s2}\n  - {path_combo}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[info] interrupted by user")
        sys.exit(130)
    except Exception as e:
        # Final guard: never dump a scary stack by default
        print(f"[fatal] Unhandled error: {e}")
        sys.exit(1)
