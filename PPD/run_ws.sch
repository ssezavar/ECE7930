python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install openai pandas

export OPENAI_API_KEY=YOUR_KEY

# CSV input (Tweets/Labels or Post,Category,Label,Sentiment,Score)
python ppd_generate_openai_plus.py --input dataset.csv --outdir artifacts --model gpt-4o-mini

# With controls
python ppd_generate_openai_plus.py --input dataset.csv --controls controls_template.csv --outdir artifacts --model gpt-4o-mini

# JSONL input
python ppd_generate_openai_plus.py --jsonl-input posts.jsonl --outdir artifacts --model gpt-4o-mini
