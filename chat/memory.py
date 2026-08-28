from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from .models import ChatSession, ChatMessage
from .llm import COVE_SYSTEM_PROMPT


def get_session_history(session: ChatSession, limit: int = 20) -> List[BaseMessage]:
    """
    Fetches the recent conversation history for a given ChatSession
    and converts the database records to LangChain BaseMessage objects.
    """
    recent_db_messages = list(session.messages.order_by("-created_at")[:limit])
    recent_db_messages.reverse()  # Chronological order

    langchain_messages: List[BaseMessage] = []

    for msg in recent_db_messages:
        if msg.role in ["assistant", "bot"]:
            langchain_messages.append(AIMessage(content=msg.content))
        elif msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))

    return langchain_messages


def save_user_message(session: ChatSession, content: str) -> ChatMessage:
    """
    Saves a user's message to the database.
    """
    return ChatMessage.objects.create(
        session=session,
        role="user",
        content=content,
    )


def save_ai_message(session: ChatSession, content: str) -> ChatMessage:
    """
    Saves an assistant's message to the database and updates session timestamp.
    """
    msg = ChatMessage.objects.create(
        session=session,
        role="assistant",
        content=content,
    )
    session.save()
    return msg
