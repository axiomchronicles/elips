import elips


database = elips.open(":memory:", dimension=768, metric="cosine")
vault = database.vault("documents")

python_history = """
Python is a high-level programming language created by Guido van Rossum.
Work on Python began in the late 1980s, and Python 0.9.0 was released in 1991.
It was designed to emphasize readability, simplicity, and developer productivity.
"""

other_documents = [
    (
        "numpy-note",
        "NumPy is a core Python library for numerical computing and multidimensional arrays.",
    ),
    (
        "database-note",
        "Vector databases are designed to store embeddings and run similarity search efficiently.",
    ),
]

history_id = vault.place_document(
    text=python_history.strip(),
    data={"topic": "python-history"},
)

for key, text in other_documents:
    vault.place_document(text=text, data={"topic": key})

query = "Who created Python and when was it first released?"
results = vault.seek_text(
    query,
    top=1,
    threshold=0.75,
)

print("Inserted history document ID:", history_id)
print("Query:", query)

if not results:
    print("\nNo similar document matched the threshold.")
else:
    best = results[0]
    print("\nBest matching document:")
    print(
        {
            "id": best.id,
            "distance": best.distance,
            "topic": best.data.get("topic"),
            "text": best.document.text,
        }
    )
