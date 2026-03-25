from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ------------------------
# 1. Chunking Strategies
# ------------------------

def semantic_chunking(text):
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


def structured_chunking(text):
    sections = text.split("\n\n")
    chunks = []
    current_header = ""

    for sec in sections:
        sec = sec.strip()
        if sec.startswith("[") and sec.endswith("]"):
            current_header = sec
        else:
            chunks.append(f"{current_header}\n{sec}")

    return chunks


# ------------------------
# 2. Choose Mode
# ------------------------

print("Select Chunking Mode:")
print("1. Semantic Chunking")
print("2. Document-aware Chunking")

choice = input("Enter choice (1 or 2): ").strip()

if choice == "1":
    mode = "semantic"
    file_path = "data/semantic.txt"
elif choice == "2":
    mode = "structured"
    file_path = "data/structured.txt"
else:
    print("Invalid choice. Defaulting to Semantic Chunking.")
    mode = "semantic"
    file_path = "data/semantic.txt"

print(f"\n✅ Selected Mode: {mode.upper()}")
print("--------------------------------------------------\n")


# ------------------------
# 3. Load Data
# ------------------------

with open(file_path, "r") as f:
    text = f.read()

if mode == "semantic":
    chunks = semantic_chunking(text)
else:
    chunks = structured_chunking(text)

docs = [Document(page_content=chunk) for chunk in chunks]


# ------------------------
# 4. Embeddings + Vector DB
# ------------------------

embedding = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma.from_documents(docs, embedding)


# ------------------------
# 5. LLM
# ------------------------

llm = OllamaLLM(model="llama3")


# ------------------------
# 6. Chat Memory
# ------------------------

chat_history = []

print("🧠 AI Assistant Ready (type 'exit' to quit)\n")


# ------------------------
# 7. Chat Loop
# ------------------------

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    # ------------------------
    # Retrieval
    # ------------------------
    results = vector_db.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in results])

    # ------------------------
    # Memory
    # ------------------------
    history_text = "\n".join(chat_history[-4:])

    print("\n🧠 MEMORY:")
    print(history_text if history_text else "None")

    print("\n📚 RAG RETRIEVAL:")
    for doc in results:
        print("-", doc.page_content)

    # ------------------------
    # Prompt
    # ------------------------
    prompt = f"""
You are an AI assistant.

Rules:
- Answer ONLY from the provided context
- If answer is not in context, say "I don't know"

Conversation:
{history_text}

Context:
{context}

Question:
{query}
"""

    # ------------------------
    # LLM Call
    # ------------------------
    response = llm.invoke(prompt)

    print("\n🤖 Answer:", response)

    print("\n📌 SOURCES:")
    for doc in results:
        print("-", doc.page_content)

    print("\n" + "=" * 60)

    # ------------------------
    # Save Memory
    # ------------------------
    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {response}")