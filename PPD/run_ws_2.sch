python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install openai pandas

# Real CSV + stratified by social_support, feeding, marital_strain
python ppd_eval_plus.py \
  --input dataset.csv \
  --synth artifacts/step2_synthetic.jsonl \
  --controls controls_template.csv \
  --strata social_support,feeding,marital_strain \
  --outdir ppd_eval_plus

# Real JSONL
python ppd_eval_plus.py \
  --jsonl-input posts.jsonl \
  --synth artifacts/step2_synthetic.jsonl \
  --outdir ppd_eval_plus_jsonl
