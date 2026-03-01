from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from configs.config import get_vector_store, get_llm


# store conversations
chat_history = []
MAX_CHAT_HISTORY = 20

# =============================
# Standalone Question Rewriter
# =============================
def rewrite_question_if_needed(user_question, model):
    """Example: Suppose the chat history is:

        User: Who is the CEO of SpaceX?
        AI: Elon Musk.

        Now the user asks:
        User: Where was he born?

        If we send "Where was he born?" directly to the retriever, it won’t know who “he” refers to.
        This function will convert it to: "Where was Elon Musk born?"

        This rewritten question can now be used safely for document retrieval."""
    if not chat_history:
        return user_question

    messages = [
        SystemMessage(content="Rewrite the new question so it is standalone.")
    ] + chat_history + [
        HumanMessage(content=f"New question: {user_question}")
    ]

    result = model.invoke(messages)
    return result.content.strip()

# =============================
# Retrieve Documents
# =============================

def retrieve_documents(query):
    db = get_vector_store()

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    return retriever.invoke(query)

# =============================
# Generate Final Answer
# =============================

def generate_answer(user_question, documents, model):
    context = "\n".join(
        [f"- {doc.page_content[:500]}" for doc in documents]
    )

    prompt = f"""
    Answer the question using ONLY the documents below.
    
    Documents:
    {context}
    
    Question: {user_question}
    
    If the answer is not in the documents, say: "I don't have enough information to answer that question."
    """

    messages = (
        [SystemMessage(content="You are a helpful assistant.")]
        + chat_history
        + [HumanMessage(content=prompt)]
    )

    result = model.invoke(messages)
    return result.content

# =============================
# Main Ask Function
# =============================

def ask_question(user_question):
    """Journey starts from here"""
    global chat_history
    model = get_llm()

    # Step 1: Rewrite question if needed
    search_query = rewrite_question_if_needed(user_question, model)

    # Step 2: Retrieve docs
    documents = retrieve_documents(search_query)

    # Step 3: Generate answer
    answer = generate_answer(user_question, documents, model)

    # Step 4: Save history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    # Step 5: Keep only last 20 messages
    chat_history = chat_history[-MAX_CHAT_HISTORY:]

    print(f"\n\nchat_history: {chat_history}")

    return answer

# =============================
# CLI Chat
# =============================

def start_chat():
    print("Chat started. Type 'quit' to exit.")

    while True:
        question = input("\nYou: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        answer = ask_question(question)
        print(f"\nAssistant: {answer}")

if __name__ == "__main__":
    start_chat()