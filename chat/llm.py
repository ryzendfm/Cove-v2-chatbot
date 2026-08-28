import re
from django.conf import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------
# Groq Configuration
# --------------------------------

GROQ_API_KEY = "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"
GROQ_MODEL   = "openai/gpt-oss-120b"


COVE_SYSTEM_PROMPT = (
    "You are Cove, a friendly, warm, intelligent, and thoughtful AI assistant and companion. "
    "You speak in an approachable, engaging, and conversational tone, like a smart and helpful friend. "
    "Do not give dry robotic responses like 'I am just a language model with no personal feelings'. "
    "Be supportive, creative, and provide well-organized answers with clean Markdown formatting when helpful."
)


def get_llm(
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    streaming: bool = True,
) -> ChatGroq:
    """
    Factory function to initialize and return a configured ChatGroq instance.
    """
    api_key = getattr(settings, "GROQ_API_KEY", GROQ_API_KEY) or GROQ_API_KEY
    selected_model = model or getattr(settings, "GROQ_MODEL", GROQ_MODEL) or GROQ_MODEL

    return ChatGroq(
        api_key=api_key,
        model=selected_model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        timeout=60,
        max_retries=3,
        reasoning_format="parsed",
    )


def generate_thread_title(user_message: str) -> str:
    """
    Generates a short, clean topic title (2 to 4 words) using LangChain.
    """
    llm = get_llm(temperature=0.2, max_tokens=15, streaming=False)

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate a very short 2 to 4 word topic title for the user request. "
            "Respond ONLY with the title without punctuation, quotes, or markdown.",
        ),
        ("human", "{user_message}"),
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    try:
        raw_title = title_chain.invoke({"user_message": user_message[:120]}).strip()
        raw_title = raw_title.split("\n")[0]
        clean_title = re.sub(r"[#*_|`=\-\[\]\(\)\{\}:;\.,\"'<>\\/!?+]+", " ", raw_title)
        clean_title = " ".join(clean_title.split())
        words = clean_title.split()
        if len(words) > 4:
            clean_title = " ".join(words[:4])
        if len(clean_title) > 28:
            clean_title = clean_title[:28].strip()
    except Exception:
        clean_title = ""

    # Fallback if empty or invalid
    if not clean_title or clean_title.lower() in [
        "title", "new chat", "topic", "here is a title", "here is the title"
    ]:
        words = [re.sub(r"\W+", "", w) for w in user_message.split() if re.sub(r"\W+", "", w)]
        clean_title = " ".join(words[:3]) if words else "Chat"

    return clean_title.title()
