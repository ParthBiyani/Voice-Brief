"""Email delivery over plain SMTP.

Works with any provider (for Gmail, use an app password). The audio is attached when it
is small enough for mailboxes to accept; above that the message says so and points at the
app instead of failing the whole send.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

from voicebrief.delivery.digest import Digest, to_html, to_text
from voicebrief.logging import get_logger

log = get_logger(__name__)

# Most providers cap a message near 25 MB once encoded (base64 adds a third).
MAX_ATTACHMENT_BYTES = 15 * 1024 * 1024


def build_message(
    digest: Digest,
    *,
    sender: str,
    recipient: str,
    audio: bytes | None,
    filename: str,
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = f"{digest.headline}: {digest.title}"[:200]

    text = to_text(digest)
    attach = audio is not None and len(audio) <= MAX_ATTACHMENT_BYTES
    if audio is not None and not attach:
        text += "\n(The audio is too large to attach; play it in the app.)\n"
    message.set_content(text)
    message.add_alternative(to_html(digest), subtype="html")

    if attach and audio is not None:
        subtype = "mpeg" if filename.endswith(".mp3") else "wav"
        message.add_attachment(audio, maintype="audio", subtype=subtype, filename=filename)
    return message


def send(
    message: EmailMessage,
    *,
    host: str,
    port: int,
    user: str | None,
    password: str | None,
) -> None:
    """Send over STARTTLS (or implicit TLS on 465)."""
    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=60) as server:
            if user and password:
                server.login(user, password)
            server.send_message(message)
        return

    with smtplib.SMTP(host, port, timeout=60) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        if user and password:
            server.login(user, password)
        server.send_message(message)
