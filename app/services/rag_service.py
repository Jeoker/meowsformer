from app.db.vector_store import get_collection
from loguru import logger
from typing import List

# Mock knowledge base data
INITIAL_KNOWLEDGE_BASE = [
    {"id": "purr_01", "text": "低频的呼噜声 (25-150Hz) 通常表示满足，但也可能用于自我治愈（如骨折愈合）。"},
    {"id": "meow_short", "text": "短促的 '喵' 声通常是打招呼或引起注意。"},
    {"id": "meow_long", "text": "拉长音的 '喵——' 可能表示要求（如要食物）或抱怨。"},
    {"id": "hiss", "text": "哈气声 (Hissing) 是防御性的警告，表示猫感到受威胁或愤怒。"},
    {"id": "growl", "text": "低沉的咆哮声 (Growling) 是进攻前的警告，表示极度愤怒或恐惧。"},
    {"id": "chirp", "text": "短促的颤音 (Chirp/Trill) 常见于母猫呼唤小猫，或猫咪看到猎物（如鸟）时的兴奋表现。"},
    {"id": "yowl", "text": "长而凄厉的嚎叫 (Yowling) 通常与发情期求偶有关，或表示身体极度不适。"},
    {"id": "scream", "text": "高频尖叫通常在打架或极度恐惧、疼痛时出现。"},
    {"id": "chatter", "text": "牙齿打颤的声音 (Chattering) 通常出现在看到无法捕获的猎物时，表示沮丧或兴奋。"},
    {"id": "silent_meow", "text": "无声的喵 (Silent Meow) 其实有高频声音，人类听不到，通常表示极度的信任和依赖。"}
]

def initialize_knowledge_base():
    """
    Check if the collection is empty, and if so, populate it with initial knowledge.
    """
    collection = get_collection()
    
    if collection.count() == 0:
        logger.info("Initializing knowledge base with default cat acoustics data...")
        
        ids = [item["id"] for item in INITIAL_KNOWLEDGE_BASE]
        documents = [item["text"] for item in INITIAL_KNOWLEDGE_BASE]
        
        collection.add(
            ids=ids,
            documents=documents
        )
        logger.info(f"Added {len(documents)} documents to the knowledge base.")
    else:
        logger.info("Knowledge base already initialized.")

def retrieve_context(query_text: str, n_results: int = 3) -> str:
    """
    Retrieve relevant context from the vector database based on the query.
    
    Args:
        query_text (str): The query text (e.g., user input or transcription).
        n_results (int): Number of results to retrieve.
        
    Returns:
        str: Concatenated string of retrieved documents.
    """
    collection = get_collection()
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # Flatten the list of lists returned by chroma
    # results['documents'] is List[List[str]]
    if results['documents']:
        retrieved_docs = results['documents'][0]
        return "\n\n".join(retrieved_docs)
    
    return ""
