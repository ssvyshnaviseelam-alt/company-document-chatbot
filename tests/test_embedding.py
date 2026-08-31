from rag.embedding import generate_embeddings


texts = [
    "Employees receive 20 vacation days per year.",
    "Employees receive 10 paid sick days per year."
]


embeddings = generate_embeddings(texts)


print("Number of embeddings:", len(embeddings))

print("Embedding dimension:", len(embeddings[0]))

print("First few values:")
print(embeddings[0][:5])