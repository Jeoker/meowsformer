import instructor
from openai import OpenAI

from app.core.config import settings


def get_openai_client() -> OpenAI:
    """Return an OpenAI client.

    Creates a new instance on every call (no internal cache).
    Callers that serve repeated requests should hold a module-level
    cached reference (e.g. via a ``_get_client()`` lazy pattern) to avoid
    re-initialising the HTTP connection pool on every request.
    """
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def get_instructor_client() -> instructor.Instructor:
    """Return an instructor-patched OpenAI client.

    Creates a new instance on every call (no internal cache).
    Callers that serve repeated requests should hold a module-level
    cached reference to avoid reconstructing the instructor wrapper on every
    call.
    """
    client = get_openai_client()
    return instructor.from_openai(client)
