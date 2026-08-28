import json
import os
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt

from urllib.parse import urlparse
from .models import ChatMessage, ChatSession, UserDocument
from .llm import generate_thread_title
from .chains import stream_chat_response
from .rag import process_document_with_progress, rebuild_user_vectorstore


# ── Page Views ───────────────────────────────────────────────────

@login_required(login_url="signin")
def index(request):
    return render(request, "chat/index.html")


def signin_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("chat_index")
        else:
            messages.error(request, "Invalid username or password.")

    return render(request, "auth/signin.html")


def signup_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        if password1 != password2:
            messages.error(request, "Passwords do not match.")
        elif User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken.")
        else:
            user = User.objects.create_user(
                username=username, email=email, password=password1
            )
            login(request, user)
            return redirect("chat_index")

    return render(request, "auth/signup.html")


def signout_view(request):
    logout(request)
    return redirect("signin")


# ── Chat API ─────────────────────────────────────────────────────

@csrf_exempt
def chat_api(request):
    """
    Streams assistant replies using Server-Sent Events (SSE) via the LangChain chains module.
    Supports dynamic RAG toggle and Tool Calling.
    """
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    user_message = data.get("message", "").strip()
    session_id = data.get("session_id")
    rag_enabled = bool(data.get("rag_enabled", False))

    if not user_message:
        return JsonResponse({"error": "Message is required"}, status=400)

    # 1. Retrieve existing session or create a new one with an AI-generated title
    session = None
    if session_id:
        try:
            session = ChatSession.objects.filter(
                id=session_id, user=request.user
            ).first()
        except Exception:
            session = None

    if not session:
        thread_title = generate_thread_title(user_message)
        session = ChatSession.objects.create(user=request.user, title=thread_title)

    # 2. Delegate SSE stream to the modular LangChain chain
    response = StreamingHttpResponse(
        stream_chat_response(session, user_message, rag_enabled=rag_enabled),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


# ── Document Upload & RAG Management APIs ─────────────────────────

@csrf_exempt
def upload_document_view(request):
    """
    Handles multi-format file uploads (.pdf, .txt, .md, .csv) and streams
    real-time vector indexing progress via SSE.
    """
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    type_map = {
        ".pdf": "pdf",
        ".txt": "text",
        ".md": "text",
        ".csv": "csv",
    }

    if ext not in type_map:
        return JsonResponse(
            {"error": "Unsupported file format. Supported formats: .pdf, .txt, .md, .csv"},
            status=400,
        )

    doc_type = type_map[ext]

    # 1. Save document to user's database records
    doc = UserDocument.objects.create(
        user=request.user,
        title=uploaded_file.name,
        doc_type=doc_type,
        file=uploaded_file,
        file_size=uploaded_file.size,
    )

    file_path = doc.file.path
    user_id = request.user.id
    doc_title = doc.title

    # 2. Generator yielding SSE progress events
    def event_stream():
        try:
            for event in process_document_with_progress(user_id, doc_type, file_path, doc_title):
                if event.get("stage") == "done":
                    # Update database model with extracted metrics
                    doc.chunk_count = event.get("chunk_count", 0)
                    doc.page_count = event.get("page_count", 0)
                    doc.save()
                    event["doc_id"] = str(doc.id)

                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            err_data = json.dumps({
                "percent": 100,
                "stage": "error",
                "message": f"Upload processing error: {str(e)}",
            })
            yield f"data: {err_data}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
def add_web_document_view(request):
    """
    Handles Web URL scraping via WebBaseLoader and streams real-time
    vector indexing progress via SSE.
    """
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
        raw_url = str(data.get("url", "")).strip()
    except Exception:
        raw_url = request.POST.get("url", "").strip()

    if not raw_url:
        return JsonResponse({"error": "URL is required"}, status=400)

    parsed = urlparse(raw_url)
    if not parsed.scheme or not parsed.netloc:
        return JsonResponse({"error": "Please enter a valid URL starting with http:// or https://"}, status=400)

    # Derive human-friendly title from domain and path
    domain = parsed.netloc.replace("www.", "")
    path_clean = parsed.path.strip("/").split("/")[-1] if parsed.path.strip("/") else ""
    title = f"{domain}/{path_clean}" if path_clean else domain

    doc = UserDocument.objects.create(
        user=request.user,
        title=title,
        doc_type="web",
        url=raw_url,
        file_size=0,
    )

    user_id = request.user.id

    def event_stream():
        try:
            for event in process_document_with_progress(user_id, "web", raw_url, title):
                if event.get("stage") == "done":
                    doc.chunk_count = event.get("chunk_count", 0)
                    doc.page_count = event.get("page_count", 1)
                    doc.save()
                    event["doc_id"] = str(doc.id)

                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            err_data = json.dumps({
                "percent": 100,
                "stage": "error",
                "message": f"Web ingestion error: {str(e)}",
            })
            yield f"data: {err_data}\n\n"

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
def documents_list_view(request):
    """
    Returns list of all documents uploaded or linked by the authenticated user.
    """
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    docs = UserDocument.objects.filter(user=request.user)
    data = [
        {
            "id": str(d.id),
            "title": d.title,
            "doc_type": d.doc_type,
            "url": d.url,
            "file_size": d.file_size,
            "chunk_count": d.chunk_count,
            "page_count": d.page_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]
    return JsonResponse({"documents": data})


@csrf_exempt
def delete_document_view(request, doc_id):
    """
    Deletes a user's document / URL and rebuilds their isolated vector store.
    """
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    if request.method != "DELETE":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    doc = get_object_or_404(UserDocument, id=doc_id, user=request.user)
    
    # Remove physical file if present
    if doc.file and os.path.exists(doc.file.path):
        try:
            os.remove(doc.file.path)
        except Exception:
            pass

    doc.delete()

    # Rebuild vector store from remaining documents
    remaining = UserDocument.objects.filter(user=request.user)
    rebuild_user_vectorstore(request.user.id, remaining)

    return JsonResponse({"success": True})


# ── Session Management APIs ──────────────────────────────────────

@csrf_exempt
def sessions_list_create(request):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    if request.method == "GET":
        sessions = ChatSession.objects.filter(user=request.user)
        data = [
            {
                "id": str(s.id),
                "title": s.title,
                "pinned": s.pinned,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ]
        return JsonResponse({"sessions": data})

    elif request.method == "POST":
        title = "New chat"
        if request.body:
            try:
                body = json.loads(request.body)
                title = body.get("title", "New chat").strip() or "New chat"
            except Exception:
                pass

        session = ChatSession.objects.create(user=request.user, title=title)
        return JsonResponse(
            {
                "id": str(session.id),
                "title": session.title,
                "pinned": session.pinned,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
            },
            status=201,
        )

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def session_detail(request, session_id):
    if not request.user.is_authenticated:
        return JsonResponse({"error": "Authentication required"}, status=401)

    session = get_object_or_404(ChatSession, id=session_id, user=request.user)

    if request.method == "GET":
        msgs = session.messages.all().order_by("created_at")
        return JsonResponse({
            "id": str(session.id),
            "title": session.title,
            "pinned": session.pinned,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "messages": [
                {
                    "id": str(m.id),
                    "role": "bot" if m.role == "assistant" else "user",
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                }
                for m in msgs
            ],
        })

    elif request.method == "PATCH":
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        if "title" in data:
            new_title = str(data["title"]).strip()
            if new_title:
                session.title = new_title
        if "pinned" in data:
            session.pinned = bool(data["pinned"])

        session.save()
        return JsonResponse({
            "id": str(session.id),
            "title": session.title,
            "pinned": session.pinned,
            "updated_at": session.updated_at.isoformat(),
        })

    elif request.method == "DELETE":
        session.delete()
        return JsonResponse({"success": True})

    return JsonResponse({"error": "Method not allowed"}, status=405)
