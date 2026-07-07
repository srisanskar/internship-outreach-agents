import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("gmail-mcp-server")


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail SMTP.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain-text email body.
    """
    gmail_address = os.environ["GMAIL_ADDRESS"]
    gmail_app_password = os.environ["GMAIL_APP_PASSWORD"]

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = to

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(gmail_address, gmail_app_password)
        server.sendmail(gmail_address, [to], msg.as_string())

    return f"Email sent successfully to {to} with subject '{subject}'."


if __name__ == "__main__":
    mcp.run(transport="stdio")
