from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="chat_index"),
    path("api/chat/", views.chat_api, name="chat_api"),
    path("api/sessions/", views.sessions_list_create, name="sessions_list_create"),
    path("api/sessions/<uuid:session_id>/", views.session_detail, name="session_detail"),

    # ── Document & RAG APIs ─────────────────────────────────────
    path("api/documents/", views.documents_list_view, name="documents_list"),
    path("api/documents/upload/", views.upload_document_view, name="upload_document"),
    path("api/documents/url/", views.add_web_document_view, name="add_web_document"),
    path("api/documents/<uuid:doc_id>/", views.delete_document_view, name="delete_document"),

    # ── Authentication ──────────────────────────────────────────
    path("signin/", views.signin_view, name="signin"),
    path("signup/", views.signup_view, name="signup"),
    path("signout/", views.signout_view, name="signout"),
]
