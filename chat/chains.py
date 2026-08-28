import json
import time
from typing import Generator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage

from .models import ChatSession
from .llm import get_llm, COVE_SYSTEM_PROMPT
from .memory import get_session_history, save_user_message, save_ai_message
from .tools import AVAILABLE_TOOLS, execute_tool_call
from .rag import get_user_retriever, format_rag_context


def stream_chat_response(
    session: ChatSession,
    user_message: str,
    rag_enabled: bool = False,
) -> Generator[str, None, None]:
    """
    Orchestrates the conversational chain with Tool Calling & Isolated User RAG support,
    yielding Server-Sent Events (SSE) data chunks for the Django view.
    """
    try:
        # 1. Yield initial event with session metadata
        init_payload = json.dumps({
            "type": "init",
            "session_id": str(session.id),
            "session_title": session.title,
        })
        yield f"data: {init_payload}\n\n"

        # 2. Retrieve conversation history and save current user message
        history = get_session_history(session, limit=20)
        save_user_message(session, user_message)

        # 3. Handle RAG Context Retrieval if enabled
        system_content = COVE_SYSTEM_PROMPT
        retrieved_sources = []

        if rag_enabled and session.user:
            retriever = get_user_retriever(session.user.id, k=5)
            if retriever is not None:
                yield f"data: {json.dumps({'type': 'status', 'status': 'Searching your documents...'})}\n\n"
                try:
                    docs = retriever.invoke(user_message)
                    if docs:
                        context = format_rag_context(docs)
                        system_content = (
                            f"{COVE_SYSTEM_PROMPT}\n\n"
                            "You have access to the user's uploaded personal documents. "
                            "Use the following retrieved context to answer the user's question accurately. "
                            "If the answer is found in the context, refer to the document and page number.\n\n"
                            f"=== RETRIEVED DOCUMENT CONTEXT ===\n"
                            f"{context}\n"
                            f"==================================="
                        )
                        for d in docs:
                            src = d.metadata.get("source_name", "Document")
                            doc_type = d.metadata.get("doc_type", "")
                            if doc_type == "pdf" and "page" in d.metadata:
                                pg = d.metadata.get("page", 0)
                                tag = f"{src} (p. {pg + 1})"
                            elif doc_type == "csv" and "row" in d.metadata:
                                row = d.metadata.get("row", 0)
                                tag = f"{src} (row {row + 1})"
                            elif doc_type == "web":
                                tag = f"{src}"
                            else:
                                tag = f"{src}"
                            if tag not in retrieved_sources:
                                retrieved_sources.append(tag)
                except Exception as rag_err:
                    print(f"RAG retrieval error: {rag_err}")

        # 4. Prepare dynamic prompt with System Message, History, and User Input
        dynamic_prompt = ChatPromptTemplate.from_messages([
            ("system", system_content),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])

        formatted_prompt = dynamic_prompt.invoke({
            "history": history,
            "input": user_message,
        })

        # 5. Initialize LLM and bind tools
        llm = get_llm(temperature=0.7, streaming=True)
        llm_with_tools = llm.bind_tools(AVAILABLE_TOOLS)

        # 6. Check if the model requests any tool calls
        first_response = llm_with_tools.invoke(formatted_prompt.messages)

        full_reply_parts = []
        thinking_open = False

        if first_response.tool_calls:
            tool_call = first_response.tool_calls[0]
            tool_name = tool_call.get("name", "tool")
            status_payload = json.dumps({
                "type": "status",
                "status": f"Running tool: {tool_name}..."
            })
            yield f"data: {status_payload}\n\n"

            tool_result = execute_tool_call(tool_call)

            followup_messages = list(formatted_prompt.messages)
            followup_messages.append(first_response)
            followup_messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

            stream_source = llm.stream(followup_messages)
        else:
            stream_source = llm.stream(formatted_prompt.messages)

        # 7. Stream tokens and reasoning chunks with <thinking> tags
        for chunk in stream_source:
            reasoning_chunk = (
                chunk.additional_kwargs.get("reasoning_content")
                if hasattr(chunk, "additional_kwargs")
                else None
            )

            if reasoning_chunk:
                if not thinking_open:
                    open_tag = "\n<thinking>\n"
                    full_reply_parts.append(open_tag)
                    yield f"data: {json.dumps({'type': 'token', 'token': open_tag})}\n\n"
                    thinking_open = True

                full_reply_parts.append(reasoning_chunk)
                yield f"data: {json.dumps({'type': 'token', 'token': reasoning_chunk})}\n\n"

            if chunk.content:
                if thinking_open:
                    close_tag = "\n</thinking>\n\n"
                    full_reply_parts.append(close_tag)
                    yield f"data: {json.dumps({'type': 'token', 'token': close_tag})}\n\n"
                    thinking_open = False

                full_reply_parts.append(chunk.content)
                yield f"data: {json.dumps({'type': 'token', 'token': chunk.content})}\n\n"

            time.sleep(0.015)

        if thinking_open:
            close_tag = "\n</thinking>\n\n"
            full_reply_parts.append(close_tag)
            yield f"data: {json.dumps({'type': 'token', 'token': close_tag})}\n\n"

        # 8. Save the final assistant response to the database
        full_reply = "".join(full_reply_parts)
        bot_msg = save_ai_message(session, full_reply)

        # 9. Yield completion event
        done_payload = json.dumps({
            "type": "done",
            "session_id": str(session.id),
            "session_title": session.title,
            "message_id": str(bot_msg.id),
            "sources": retrieved_sources,
        })
        yield f"data: {done_payload}\n\n"

    except Exception as e:
        err_payload = json.dumps({"type": "error", "error": str(e)})
        yield f"data: {err_payload}\n\n"
