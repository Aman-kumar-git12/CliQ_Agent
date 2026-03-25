from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from core.llm import llm
from rag.prompt import RAG_PROMPT, ROUTER_PROMPT
from rag.vectorstore import load_vectorstore


def get_rag_chain():
    vectorstore = load_vectorstore()

    # 1. Contextualize Question (History-aware)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create the standalone question generator
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    # 2. Router: Identify the target file
    router_chain = ROUTER_PROMPT | llm | StrOutputParser()

    # 3. Dynamic Filtering Retriever Logic
    def route_and_retrieve(info):
        standalone_question = info["standalone_question"]
        chat_history = info["chat_history"]
        
        # Determine the file using the router
        target_file = router_chain.invoke({"input": standalone_question, "chat_history": chat_history})
        target_file = target_file.strip().lower()
        
        # Extract filename from potential prose (just in case LLM is wordy)
        if ".txt" not in target_file:
            # Fallback or simple search if LLM fails to provide just the filename
            target_file = "cliq_kb.txt"
        
        print(f"Routing Decision: target_file='{target_file}' for query='{standalone_question}'")
        
        # Apply pre-filter for MongoDB Atlas Vector Search
        search_kwargs = {"k": 3}
        if ".txt" in target_file:
            search_kwargs["pre_filter"] = {"source": {"$eq": target_file}}
            
        docs = vectorstore.similarity_search(standalone_question, **search_kwargs)
        return docs

    # Document chain for answering
    document_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
    
    # Construct the full intelligent chain
    # It MUST return a dictionary with "answer" and "context" keys to match chat.py usage
    full_chain = (
        RunnablePassthrough.assign(
            standalone_question=contextualize_chain
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(route_and_retrieve)
        )
        | RunnablePassthrough.assign(
            answer=document_chain
        )
    )

    return full_chain