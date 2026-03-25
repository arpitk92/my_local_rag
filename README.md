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


def bad_chunking(text):
    # intentionally bad: break text arbitrarily into small chunks
    words = text.split()
    chunk_size = 8  # small → breaks meaning
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ------------------------
# 2. Choose Mode
# ------------------------

print("Select Chunking Mode:")
print("1. Semantic Chunking")
print("2. Document-aware Chunking")
print("3. Bad Chunking (Failure Mode)")

choice = input("Enter choice (1, 2 or 3): ").strip()

if choice == "1":
    mode = "semantic"
    file_path = "data/semantic.txt"

elif choice == "2":
    mode = "structured"
    file_path = "data/structured.txt"

elif choice == "3":
    mode = "bad"
    file_path = "data/semantic.txt"  # use same data to compare fairly

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

elif mode == "structured":
    chunks = structured_chunking(text)

elif mode == "bad":
    chunks = bad_chunking(text)

# Debug: show chunk count
print(f"📦 Total chunks created: {len(chunks)}\n")

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