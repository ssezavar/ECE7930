#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppd_generate_ollama.py
Drop-in generator for PPD synthetic data with FREE providers by default.

Provider priority (auto):
  1) Ollama (local, free)       --provider=ollama   --ollama-model mistral
  2) Hugging Face Inference API --provider=hf       --hf-model HuggingFaceH4/zephyr-7b-beta
  3) OpenAI                     --provider=openai   --openai-model gpt-4o-mini
  4) Mock (rule-based)          --provider=mock or --dry-run

Inputs:
  python ppd_generate_ollama.py --input dataset.csv --outdir artifacts --provider auto --max-rows 3000

Outputs (in --outdir):
  - step1_synopses.jsonl
  - step2_synthetic.jsonl
  - combined_preview.csv
"""

import argparse, csv, hashlib, json, os, random, re, sys, time, shutil, subprocess
from typing import Dict, List, Tuple, Optional

# ------------------------------- CLI & Config ----------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPD synthetic data generator (Ollama/HF/OpenAI)")
    p.add_argument("--input", required=True, help="Path to input CSV (expects columns: Post, Label, Category, Sentiment)")
    p.add_argument("--outdir", default="artifacts", help="Directory for outputs (default: artifacts)")
    p.add_argument("--provider", default="auto", choices=["auto","ollama","hf","openai","mock"], help="Provider to use (default: auto)")
    p.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model (if provider=openai)")
    p.add_argument("--hf-model", default="HuggingFaceH4/zephyr-7b-beta", help="HF model id (if provider=hf)")
    p.add_argument("--ollama-model", default="mistral", help="Ollama model name (if provider=ollama)")
    p.add_argument("--dry-run", action="store_true", help="Force mock outputs (overrides provider)")
    p.add_argument("--max-rows", type=int, default=None, help="Limit rows for quick tests")
    p.add_argument("--dedup-threshold", type=float, default=0.90, help="Jaccard overlap threshold to flag duplicates")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--column-map", default=None, help="Optional mapping like: post=Text,label=Target,category=Topic,sentiment=Polarity")
    return p.parse_args()

# ------------------------------- Utilities -------------------------------------
def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _resolve_columns(fieldnames, user_map: Optional[Dict[str,str]]=None):
    fnorm = { (f or "").strip().lower(): f for f in (fieldnames or []) }
    resolved = {}
    if user_map:
        for need, have in user_map.items():
            if have in fieldnames:
                resolved[need] = have
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
        resolved, missing = _resolve_columns(r.fieldnames, user_map=user_map)
        if missing:
            raise ValueError(
                "Input CSV missing required columns. "
                f"Need logical columns: ['Post','Label','Category','Sentiment']\n"
                f"Your header: {r.fieldnames}\n"
                "Tip: rename headers OR pass --column-map post=YourPost,label=YourLabel,category=YourCategory,sentiment=YourSentiment"
            )
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

# ------------------------------- Providers --------------------------------------
def have_ollama() -> bool:
    return shutil.which("ollama") is not None

def gen_ollama(model: str, prompt: str, temperature: float=0.6, max_tokens: int=220) -> str:
    # Send prompt via stdin to avoid shell quoting issues
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        return out
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")

def have_hf() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False

def gen_hf(model_id: str, prompt: str, temperature: float=0.6, max_tokens: int=220) -> str:
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model_id)
        # Many instruct models accept plain prompt; some expect chat template
        text = client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"HF error: {e}")

def have_openai() -> bool:
    try:
        from openai import OpenAI  # noqa: F401
        return True
    except Exception:
        return False

def gen_openai(model: str, prompt: str, temperature: float=0.6, max_tokens: int=220) -> str:
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI error: {e}")

def detect_provider(pref: str) -> str:
    if pref == "mock":
        return "mock"
    if pref == "ollama":
        return "ollama" if have_ollama() else "mock"
    if pref == "hf":
        return "hf" if have_hf() else ("ollama" if have_ollama() else "mock")
    if pref == "openai":
        return "openai" if have_openai() else ("ollama" if have_ollama() else "mock")
    # auto
    if have_ollama():
        return "ollama"
    if have_hf():
        return "hf"
    if have_openai():
        return "openai"
    return "mock"

def gen_text(provider: str, model_openai: str, model_hf: str, model_ollama: str, prompt: str, temperature: float, max_tokens: int) -> str:
    if provider == "ollama":
        return gen_ollama(model_ollama, prompt, temperature, max_tokens)
    if provider == "hf":
        return gen_hf(model_hf, prompt, temperature, max_tokens)
    if provider == "openai":
        return gen_openai(model_openai, prompt, temperature, max_tokens)
    # mock fallback
    return ""

# ------------------------------- Prompts ----------------------------------------
def prompt_step1(post: str, category: str, label: str, sentiment: str) -> str:
    return f"""
You are assisting with postpartum mental health data abstraction (no PHI).
Summarize the user's post into a brief, clinician-friendly synopsis and list 2–4 salient contextual factors.
Then estimate a coarse EPDS risk bucket label (do NOT output a number), choose one of:
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
""".strip()

def prompt_step2(s1: Dict, target: str) -> str:
    return f"""
Write a short, empathetic, first-person postpartum diary entry (100–180 words).
Use the synopsis and salient factors below. Avoid PHI or realistic identifiers.
The tone should be human, not clinical, and align with the guidance target.

SYNOPSIS: {json.dumps(s1, ensure_ascii=False)}
TARGET: {target}

Return ONLY the diary text, no preface or JSON.
""".strip()

# ------------------------------- Main Pipeline ----------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    ensure_outdir(args.outdir)

    provider = detect_provider(args.provider)
    if args.dry_run:
        provider = "mock"

    print("=" * 72)
    print(f"[provider] {provider.upper()}")
    if provider == "ollama":
        print(f"  - model: {args.ollama-model if False else args.ollama_model}")
    elif provider == "hf":
        print(f"  - model: {args.hf_model}")
    elif provider == "openai":
        print(f"  - model: {args.openai_model}")
    else:
        print("  - mock rule-based generator")
    print("=" * 72)

    # Output paths
    path_s1 = os.path.join(args.outdir, "step1_synopses.jsonl")
    path_s2 = os.path.join(args.outdir, "step2_synthetic.jsonl")
    path_combo = os.path.join(args.outdir, "combined_preview.csv")

    rows = read_rows(args.input, max_rows=args.max_rows, column_map=args.column_map)

    step1_records: List[Dict] = []
    step2_records: List[Dict] = []
    combined_rows: List[Dict[str, str]] = []

    seen_s1_tokens: List[List[str]] = []
    seen_s2_tokens: List[List[str]] = []

    for idx, row in enumerate(rows):
        post = row["Post"]; category = row["Category"]; label = row["Label"]; sentiment = row["Sentiment"]
        s1 = None; s2 = None; skip_reason = ""; used_mock = (provider == "mock")

        try:
            # ---- STEP 1 ----
            if provider == "mock":
                s1 = mock_step1_output(post, category, label, sentiment)
            else:
                p1 = prompt_step1(post, category, label, sentiment)
                try:
                    content = gen_text(provider, args.openai_model, args.hf_model, args.ollama_model, p1, temperature=0.2, max_tokens=260)
                    # Try to parse JSON
                    parsed = json.loads(content) if content else {}
                    summary = (parsed.get("summary") or "").strip()
                    salient = parsed.get("salient_factors") or []
                    risk = (parsed.get("estimated_risk") or "").strip()
                    if not summary:
                        raise ValueError("empty summary from provider")
                    s1 = {
                        "type": "synopsis",
                        "summary": summary,
                        "salient_factors": salient if isinstance(salient, list) else [],
                        "estimated_risk": risk or choose_epds_target(label, sentiment),
                        "source_hash": hashlib.md5(post.encode("utf-8")).hexdigest(),
                    }
                except Exception as e:
                    # fallback per-row
                    s1 = mock_step1_output(post, category, label, sentiment)
                    used_mock = True

            # Dedup STEP 1
            if is_duplicate(s1.get("summary",""), seen_s1_tokens, args.dedup_threshold):
                skip_reason = "dup_step1"
            else:
                seen_s1_tokens.append(normalize_text(s1.get("summary","")))
                step1_records.append(s1)

            # ---- STEP 2 ----
            target = s1.get("estimated_risk") or choose_epds_target(label, sentiment)
            if provider == "mock":
                s2 = mock_step2_output(s1, target)
            else:
                p2 = prompt_step2(s1, target)
                try:
                    text = gen_text(provider, args.openai_model, args.hf_model, args.ollama_model, p2, temperature=0.6, max_tokens=260)
                    if not text:
                        raise ValueError("empty diary text")
                    s2 = {
                        "type": "synthetic_post",
                        "synthetic_text": text.strip(),
                        "target": target,
                        "style": "first-person postpartum diary",
                    }
                except Exception as e:
                    s2 = mock_step2_output(s1, target)
                    used_mock = True

            # Dedup STEP 2
            if is_duplicate(s2.get("synthetic_text",""), seen_s2_tokens, args.dedup_threshold):
                skip_reason = (skip_reason + "|dup_step2").strip("|")
            else:
                seen_s2_tokens.append(normalize_text(s2.get("synthetic_text","")))
                step2_records.append(s2)

        except Exception as fatal:
            skip_reason = (skip_reason + "|fatal").strip("|")

        # Combined preview row
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

        if (idx+1) % 25 == 0:
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
        print("\n[info] interrupted by user"); sys.exit(130)
    except Exception as e:
        print(f"[fatal] Unhandled error: {e}"); sys.exit(1)
