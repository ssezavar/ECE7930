# v7.3 — Clinical-Strict (best for EPDS label accuracy)

import argparse, json, csv, os, time, subprocess, smtplib
from email.message import EmailMessage
from multiprocessing import Pool, Manager

BATCH_SIZE = 20
RETRIES = 3
RETRY_DELAY = 2
LOG_FILE = "batch_log.txt"
CHECKPOINT_FILE = "checkpoint_done.txt"


# ---------------- Logging ----------------

def log(msg: str):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Logging should never crash the job
        pass


# ---------------- LLM with retries ----------------

def run_ollama(
    model,
    prompt,
    temperature,
    top_p,
    top_k,
    repeat_penalty,
    max_tokens,
    max_retries: int = 4,
    cooldown_base: float = 1.8,
) -> str:
    """
    Call Ollama with JSON payload + sampling options.
    Includes simple auto-throttling:
      - backs off on empty / very short outputs
      - backs off on timeouts
    """

    payload = {
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "num_predict": max_tokens,
        },
    }

    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            proc = subprocess.run(
                ["ollama", "run", model],
                input=json.dumps(payload).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
            )
            out = proc.stdout.decode("utf-8", errors="ignore").strip()

            # --- auto-throttling heuristics ---

            # 1) Completely empty → treat as overload, back off
            if not out:
                sleep_time = cooldown_base * attempt
                log(f"Ollama empty output (attempt {attempt}), sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                continue

            # 2) Suspiciously short → likely truncated / error text, back off lightly
            if len(out) < 40:
                sleep_time = cooldown_base * 0.5 * attempt
                log(f"Ollama very short output (len={len(out)}), sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                # still return it; JSON layer will decide if usable
                return out

            # 3) If response came *too* fast, add a tiny delay to avoid bursts
            elapsed = time.time() - start
            if elapsed < 0.3:
                time.sleep(0.1)

            return out

        except subprocess.TimeoutExpired:
            sleep_time = cooldown_base * (attempt + 1)
            log(f"Ollama timeout on attempt {attempt}, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        except Exception as e:
            sleep_time = cooldown_base * attempt
            log(f"Ollama error on attempt {attempt}: {e}. Sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

    log("Ollama failed after all retries, returning empty string.")
    return ""





# ---------------- Prompts ----------------

def prompt_step1(batch):
    return (
        "You are an expert in postpartum mental health.\n"
        "Your task is to analyze each entry and produce:\n"
        "  • summary: a concise clinical-style summary (2–3 sentences)\n"
        "  • salient_factors: list of factors contributing to distress\n"
        "  • estimated_risk: one of these exact labels:\n"
        "        'EPDS 0–4 (low risk)',\n"
        "        'EPDS 5–9 (watchful waiting)',\n"
        "        'EPDS 10–12 (mild PPD)',\n"
        "        'EPDS 13–19 (moderate PPD)',\n"
        "        'EPDS 20+ (severe PPD)'\n"
        "\n"
        "IMPORTANT:\n"
        "• Output ONLY a JSON LIST.\n"
        "• Length and order must match input exactly.\n\n"
        "INPUT:\n" + json.dumps(batch, ensure_ascii=False)
    )


# v7.3 — Clinical-Strict (best for EPDS label accuracy)
def prompt_step2(step1_list):
    return (
        "You are generating structured first-person postpartum diaries where emotional content and\n"
        "symptom presentation MUST precisely match the EPDS classification.\n"
        "\n"
        "Rules:\n"
        "• Length 150–220 words.\n"
        "• Emotional descriptions must mirror the target EPDS risk level.\n"
        "• Avoid poetic language. Use clear, grounded, clinical-feeling emotional clarity.\n"
        "• Manually emphasize indicators consistent with the risk category (without naming EPDS).\n"
        "• Keep text functional, moderately varied, but not overly creative.\n"
        "\n"
        "Output JSON LIST with fields: synthetic_text, target, style='clinical-strict diary'.\n\n"
        "INPUT:\n" + json.dumps(step1_list, ensure_ascii=False)
    )


# ---------------- Dedup ----------------

def normalize(s: str) -> str:
    return " ".join((s or "").lower().split())


def dedup_list(rows):
    """
    Dedup within a batch by s2_text (synthetic diary text).
    rows: list of combined-row dicts.
    """
    seen = set()
    result = []
    for item in rows:
        text = item.get("s2_text", "")
        key = normalize(text)
        if not key:
            # Keep empty / weird rows; they are rare
            result.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


# ---------------- Process Batch ----------------

def process_batch(args):
    (
        batch_id,
        batch_rows,
        model,
        progress,
        cfg,
    ) = args

    log(f"Batch {batch_id}: starting ({len(batch_rows)} rows)")

    # Build payload for Step 1
    items = []
    for i, r in enumerate(batch_rows):
        items.append({
            "id": i,
            "post": r.get("Post", ""),
            "label": r.get("Label", ""),
            "category": r.get("Category", ""),
            "sentiment": r.get("Sentiment", ""),
        })

    # ---- STEP 1 ----
    out1 = run_ollama(
        model,
        prompt_step1(items),
        temperature=cfg["temp_step1"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        repeat_penalty=cfg["repeat_penalty"],
        max_tokens=cfg["max_tokens_step1"],
    )

    s1_list = None
    if out1:
        try:
            parsed = json.loads(out1)
            if isinstance(parsed, list) and len(parsed) == len(batch_rows):
                s1_list = parsed
            else:
                log(f"Batch {batch_id}: step1 JSON length mismatch or not a list. Using fallback.")
        except Exception as e:
            log(f"Batch {batch_id}: step1 JSON parse error: {e}")

    if s1_list is None:
        # Fallback: simple summaries from the original posts
        s1_list = []
        for r in batch_rows:
            s1_list.append({
                "summary": r.get("Post", "")[:150],
                "salient_factors": [],
                "estimated_risk": "EPDS 5–9 (watchful waiting) – psychoeducation",
            })

    # ---- STEP 2 ----
    out2 = run_ollama(
        model,
        prompt_step2(s1_list),
        temperature=cfg["temp_step2"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        repeat_penalty=cfg["repeat_penalty"],
        max_tokens=cfg["max_tokens_step2"],
    )

    s2_list = None
    if out2:
        try:
            parsed2 = json.loads(out2)
            if isinstance(parsed2, list) and len(parsed2) == len(s1_list):
                s2_list = parsed2
            else:
                log(f"Batch {batch_id}: step2 JSON length mismatch or not a list. Using fallback.")
        except Exception as e:
            log(f"Batch {batch_id}: step2 JSON parse error: {e}")

    if s2_list is None:
        # Fallback: re-use summaries as synthetic text
        s2_list = []
        for s1 in s1_list:
            s2_list.append({
                "synthetic_text": s1.get("summary", ""),
                "target": s1.get("estimated_risk", ""),
                "style": "first-person postpartum diary",
            })

    # Build combined rows
    combo = []
    for i, r in enumerate(batch_rows):
        s1 = s1_list[i]
        s2 = s2_list[i]
        combo.append({
            "post": r.get("Post", ""),
            "label": r.get("Label", ""),
            "sentiment": r.get("Sentiment", ""),
            "category": r.get("Category", ""),
            "s1_summary": s1.get("summary", ""),
            "s1_estimated_risk": s1.get("estimated_risk", ""),
            "s2_text": s2.get("synthetic_text", ""),
            "s2_target": s2.get("target", ""),
            "s2_style": s2.get("style", ""),
        })

    # Dedup inside this batch
    before = len(combo)
    combo = dedup_list(combo)
    after = len(combo)
    if after < before:
        log(f"Batch {batch_id}: dedup removed {before - after} rows inside batch.")

    # Mark progress
    if progress is not None:
        try:
            progress.value += 1    # Python 3.12 ValueProxy (no lock)
        except:
            pass

    # Save checkpoint
    try:
        with open(CHECKPOINT_FILE, "a", encoding="utf-8") as ck:
            ck.write(f"{batch_id}\n")
    except Exception as e:
        log(f"Batch {batch_id}: failed to write checkpoint: {e}")

    log(f"Batch {batch_id}: finished with {after} rows.")
    return combo


# ---------------- Progress bar ----------------

def print_progress(progress, total):
    done = progress.value
    if total <= 0:
        print("\r[----------------------------------------] 0/0 batches", end="", flush=True)
        return
    pct = done / total
    bar_len = 40
    fill = int(bar_len * pct)
    bar = "#" * fill + "-" * (bar_len - fill)
    print(f"\r[{bar}] {done}/{total} batches", end="", flush=True)


# ---------------- Email ----------------

def send_email():
    user = os.getenv("GMAIL_USER")
    pwd = os.getenv("GMAIL_PASS")
    if not user or not pwd:
        log("Email skipped: GMAIL_USER or GMAIL_PASS missing.")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = "PPD Batch Run Finished"
        msg["From"] = user
        msg["To"] = user
        msg.set_content("Your AWS batch run has completed successfully.")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(user, pwd)
            server.send_message(msg)

        log("Email sent.")
    except Exception as e:
        log(f"Email failed: {e}")


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV (expects Post, Label, Category, Sentiment)",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--ollama-model",
        required=True,
        help="Ollama model name, e.g. mistral:instruct",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )

    # Tuning controls
    parser.add_argument(
        "--temp-step1",
        type=float,
        default=0.9,
        help="Temperature for Step 1 (summaries). Default=0.9",
    )
    parser.add_argument(
        "--temp-step2",
        type=float,
        default=1.2,
        help="Temperature for Step 2 (diary generation). Default=1.2",
    )
    parser.add_argument(
        "--max-tokens-step1",
        type=int,
        default=220,
        help="Max tokens for Step 1 outputs. Default=220",
    )
    parser.add_argument(
        "--max-tokens-step2",
        type=int,
        default=260,
        help="Max tokens for Step 2 diary outputs. Default=260",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling. Default=0.9",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling. Default=50",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.15,
        help="Repeat penalty. Default=1.15",
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log("Starting batch run...")

    # Load data
    rows = []
    with open(args.input, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        log("No rows found in input CSV. Exiting.")
        return

    # Build batches: (batch_id, batch_rows, model)
    batches = []
    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i:i + BATCH_SIZE]
        batch_id = i // BATCH_SIZE
        batches.append((batch_id, batch_rows, args.ollama_model))

    # Load checkpoint
    done_batches = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as ck:
            for line in ck:
                line = line.strip()
                if line:
                    try:
                        done_batches.add(int(line))
                    except ValueError:
                        pass

    # Filter out already-done batches
    unfinished = []
    for batch_id, batch_rows, model in batches:
        if batch_id not in done_batches:
            unfinished.append((batch_id, batch_rows, model))

    total_batches = len(unfinished)
    log(f"Total batches to run: {total_batches}")

    if total_batches == 0:
        log("All batches already completed.")
        send_email()
        log("Shutting down...")
        os.system("sudo shutdown -h now")
        return

    # Shared progress counter
    manager = Manager()
    progress = manager.Value("i", 0)

    # Config for tuning
    cfg = {
        "temp_step1": args.temp_step1,
        "temp_step2": args.temp_step2,
        "max_tokens_step1": args.max_tokens_step1,
        "max_tokens_step2": args.max_tokens_step2,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
    }

    # Prepare args for workers: (batch_id, batch_rows, model, progress, cfg)
    batch_args = [
        (batch_id, batch_rows, model, progress, cfg)
        for (batch_id, batch_rows, model) in unfinished
    ]

    # ------------------ OLLAMA MODEL WARMUP (ADD THIS) ------------------
    log("Warming up model...")
    _ = run_ollama(
        args.ollama_model,
        "Hello",                # tiny prompt to load the model into VRAM
        temperature=0.1,
        top_p=1.0,
        top_k=50,
        repeat_penalty=1.0,
        max_tokens=20,
    )
    log("Warmup complete.")
    time.sleep(2)
    # ---------------------------------------------------------------------

    # Run workers
    log("Processing...")
    with Pool(args.workers) as pool:
        async_res = pool.map_async(process_batch, batch_args)


        while not async_res.ready():
            print_progress(progress, total_batches)
            time.sleep(0.5)

        results = async_res.get()

    print_progress(progress, total_batches)
    print()

    # Flatten results
    combined = [item for batch in results for item in batch]

    if not combined:
        log("Warning: combined result is empty; nothing to write.")
    else:
        outpath = os.path.join(args.outdir, "combined_batch.csv")
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(combined[0].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(combined)
        log(f"Done. Output written: {outpath}")

    # Email + shutdown
    send_email()
    log("Shutting down...")
    os.system("sudo shutdown -h now")


if __name__ == "__main__":
    main()
