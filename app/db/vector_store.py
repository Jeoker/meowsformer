import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

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
