import chromadb
from chromadb.utils import embedding_functions
from app.core.config import settings

# Initialize ChromaDB persistent client
# By default, Chroma will use the ./db/chroma_db directory specified in config.py
client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

# Use OpenAI's text-embedding-3-small model
# Note: This requires the OPENAI_API_KEY to be set in the environment.
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=settings.OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

def get_collection():
    """
    Get or create the 'cat_acoustics' collection.
    """
    return client.get_or_create_collection(
        name="cat_acoustics",
        embedding_function=openai_ef
    )
