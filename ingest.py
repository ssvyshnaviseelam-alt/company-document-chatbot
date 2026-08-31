from pathlib import Path
import pickle

import faiss
import numpy as np
from pypdf import PdfReader

from rag.embedding import generate_embeddings


# --------------------------------
# 1. Load PDF documents
# --------------------------------

data_folder = Path("data")

documents = []

for pdf_file in data_folder.glob("*.pdf"):

    reader = PdfReader(pdf_file)

    print(f"\nReading: {pdf_file.name}")
    print(f"Pages: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            documents.append({
                "text": text,
                "source": pdf_file.name,
                "page": page_number + 1
            })


print(f"\nTotal pages loaded: {len(documents)}")


# --------------------------------
# 2. Split documents into chunks
# --------------------------------

chunk_size = 500
chunk_overlap = 100

chunks = []

for document in documents:

    text = document["text"]

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "source": document["source"],
            "page": document["page"]
        })

        start = end - chunk_overlap


print(f"Total chunks created: {len(chunks)}")


# --------------------------------
# 3. Display a few chunks
# --------------------------------

for i, chunk in enumerate(chunks[:3]):

    print("\n-----------------------------")
    print(f"Chunk {i + 1}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")
    print("-----------------------------")

    print(chunk["text"])


# --------------------------------
# 4. Generate Gemini embeddings
# --------------------------------

print("\nGenerating embeddings using Gemini...")

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = generate_embeddings(texts)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

print(f"Embeddings created: {embeddings.shape}")


# --------------------------------
# 5. Create FAISS index
# --------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"Vectors stored in FAISS: {index.ntotal}")


# --------------------------------
# 6. Save FAISS index
# --------------------------------

faiss_folder = Path("faiss_index")

faiss_folder.mkdir(
    exist_ok=True
)

faiss.write_index(
    index,
    str(faiss_folder / "index.faiss")
)


# --------------------------------
# 7. Save chunks
# --------------------------------

with open(
    faiss_folder / "chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        chunks,
        f
    )


print("\nFAISS index saved successfully!")

print(
    "Saved index: "
    "faiss_index/index.faiss"
)

print(
    "Saved chunks: "
    "faiss_index/chunks.pkl"
)