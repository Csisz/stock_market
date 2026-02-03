import os
import smtplib
from email.message import EmailMessage

def send_email(to_email: str, subject: str, body: str, html_body: str | None = None):
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "465"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")
    from_email = os.getenv("FROM_EMAIL", user)

    if not all([host, user, password, from_email]):
        raise RuntimeError("Missing SMTP env vars (SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/FROM_EMAIL)")

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    # Plain text kötelező
    msg.set_content(body)

    # HTML opcionális
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL(host, port) as s:
        s.login(user, password)
        s.send_message(msg)
