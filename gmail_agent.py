import os
import base64
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from dateutil import parser as date_parser

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


# =========================
# CONFIG
# =========================

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000,
    streaming=True
)


# =========================
# GMAIL AUTH
# =========================

def authenticate_gmail():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# =========================
# TOOL 1: READ + SUMMARIZE
# =========================

class ReadEmailsInput(BaseModel):
    date: str = Field(description="Natural language date like Feb 11th 2026")
    sender: str | None = Field(default=None, description="Email address of sender to filter")


def read_and_summarize(date: str,sender: str | None = None):
    service = authenticate_gmail()

    date_obj = date_parser.parse(date)
    next_day = date_obj + timedelta(days=1)

    query = f"after:{date_obj.strftime('%Y/%m/%d')} before:{next_day.strftime('%Y/%m/%d')}"

    if sender:
        query += f" from:{sender}"

    results = service.users().messages().list(
        userId="me",
        q=query
    ).execute()

    messages = results.get("messages", [])[:12]

    if not messages:
        return "No emails found for that date."

    

    full_text = ""

    for msg in messages:
        msg_data = service.users().messages().get(
            userId="me",
            id=msg["id"],
            format="full"
        ).execute()

        payload = msg_data.get("payload", {})
        parts = payload.get("parts", [])

        # Handle emails with parts (multipart)
        if parts:
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    data = part["body"].get("data")
                    if data:
                        decoded = base64.urlsafe_b64decode(data).decode()
                        decoded = decoded[:3000]
                        full_text += decoded + "\n"
        # Handle emails without parts (simple text email)
        else:
            if payload.get("mimeType") == "text/plain":
                data = payload.get("body", {}).get("data")
                if data:
                    decoded = base64.urlsafe_b64decode(data).decode()
                    decoded = decoded[:3000]
                    full_text += decoded + "\n"

    summary = llm.invoke(f"Summarize these emails clearly:\n\n{full_text}").content

    return summary


read_tool = StructuredTool.from_function(
    func=read_and_summarize,
    name="ReadAndSummarizeEmails",
    description="Reads emails from a specific date and summarizes them",
    args_schema=ReadEmailsInput,
)


# =========================
# TOOL 2: SEND EMAIL
# =========================

class SendEmailInput(BaseModel):
    to: str
    subject: str
    body: str


def send_email(to: str, subject: str, body: str):
    service = authenticate_gmail()

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    service.users().messages().send(
        userId="me",
        body={"raw": raw}
    ).execute()

    return "Email sent successfully."


send_tool = StructuredTool.from_function(
    func=send_email,
    name="SendEmail",
    description="Send an email to a recipient",
    args_schema=SendEmailInput,
)


# =========================
# AGENT SETUP
# =========================

tools = [read_tool, send_tool]

# Create the agent using LangChain 1.2.10 API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that can read and send emails.",
    debug=True
)

if __name__ == "__main__":

    user_prompt = "Send the summarized content which have received on Feb 12th 2026  to *****@gmail.com"

    # Stream the results using LangGraph API
    for chunk in agent.stream({"messages": [HumanMessage(content=user_prompt)]}):
        print(chunk)

