import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

# Configure embedding function based on API provider.
# text-embedding-3-small is supported by both OpenAI and ai-builders.
if settings.API_PROVIDER == "ai_builders":
    _embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.AI_BUILDER_TOKEN,
        api_base=settings.AI_BUILDER_BASE_URL,
        model_name="text-embedding-3-small",
    )
else:
    _embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )


def get_collection():
    """
    Get or create the 'cat_acoustics' collection.
    """
    return client.get_or_create_collection(
        name="cat_acoustics",
        embedding_function=_embedding_fn,
    )
