"""
Multi-document QA example — HotpotQA style.

Simulates a typical use case: given a question and a set of candidate
documents, rank the most relevant ones and pass only those to your LLM.
"""

from doc2graph import DocumentGraph

# Simulate a small document corpus (like HotpotQA context)
documents = [
    {
        "title": "Python (programming language)",
        "content": (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability. "
            "Python was created by Guido van Rossum and first released in 1991."
        ),
    },
    {
        "title": "Guido van Rossum",
        "content": (
            "Guido van Rossum is a Dutch programmer best known as the creator of Python. "
            "He was born in 1956 in the Netherlands. "
            "He served as Python's 'benevolent dictator for life' until 2018."
        ),
    },
    {
        "title": "Java (programming language)",
        "content": (
            "Java is a high-level, class-based, object-oriented programming language. "
            "Java was originally developed by James Gosling at Sun Microsystems."
        ),
    },
    {
        "title": "Netherlands",
        "content": (
            "The Netherlands is a country in Northwestern Europe. "
            "Amsterdam is its capital. The Dutch are known for their engineering."
        ),
    },
    {
        "title": "James Gosling",
        "content": (
            "James Gosling is a Canadian computer scientist, best known as the creator of Java. "
            "He was born in Calgary, Alberta, Canada."
        ),
    },
]

# Build a document graph
g = DocumentGraph.from_texts(documents)
print(f"Graph: {g}")
print()

# Multi-hop question — requires Python + Guido van Rossum
question = "In what country was the creator of Python born?"
print(f"Question: {question}")
print()

context = g.rank(question, k=3)

print(f"Top-{len([n for n in context['nodes'] if 'score' in n])} ranked nodes:")
for node in sorted(context["nodes"], key=lambda n: n.get("score", 0), reverse=True):
    score = node.get("score")
    marker = f"  [score={score:.4f}]" if score else "  [1-hop neighbour]"
    print(f"  - {node['label']}{marker}")

print()
print("Context edges:")
for edge in context["edges"]:
    print(f"  {edge['from']} --[{edge['label']}]--> {edge['to']}")

print()
print("Pass context['nodes'] + context['edges'] to your LLM for focused, token-efficient QA.")
