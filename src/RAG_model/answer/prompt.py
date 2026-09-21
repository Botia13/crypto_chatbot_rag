"""Pure prompt construction shared by the app and offline-context benchmarks."""

def make_rag_messages(question, history, chunks, system_prompt: str):
    """Build chat messages for the RAG answer step: system (with context) + history + user question."""
    
    context = "\n\n".join(
        f"""[SOURCE: {chunk['chunk_id']}]
    Ticker: {chunk['ticker']}
    Section: {chunk['section_title']}
    Section key: {chunk.get('section_key', 'N/A')}
    Table: {chunk.get('table_title') or 'N/A'}
    SEC URL: {chunk['source_url']}

    Content:
    {chunk['chunk_text']}"""
        for chunk in chunks)
    
    system_prompt = f"""

    {system_prompt}

    Context:
    {context}

    """

    
    return (
        [{"role": "system",
          "content": system_prompt}]
        + history
        + [{"role": "user",
            "content": question}]
        )
