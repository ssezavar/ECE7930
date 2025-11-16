tmux new-session -d -s ppd_gpu "

python3 ppd_generate_ollama_v2.py \
  --input dataset.csv \
  --outdir artifacts_v4_ollama2 \
  --provider ollama \
  --ollama-model mistral:instruct

python3 - << 'EOF'
import os, ssl, smtplib
from email.message import EmailMessage

try:
    user = os.environ.get('GMAIL_USER')
    pwd  = os.environ.get('GMAIL_PASS')

    if user and pwd:
        msg = EmailMessage()
        msg['Subject'] = 'PPD GPU run finished'
        msg['From'] = user
        msg['To'] = 'sara.sezavar1@gmail.com'
        msg.set_content('The AWS GPU PPD synthetic data run has finished.')

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(user, pwd)
            server.send_message(msg)
except:
    pass
EOF

sudo shutdown -h now
"
