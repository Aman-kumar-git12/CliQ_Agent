import json

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from core.llm import classifier_llm, response_llm
from rag.prompt import DIRECT_RESPONSE_PROMPT, INTENT_PROMPT, RAG_PROMPT
from rag.vectorstore import CliQVectorStore


class CliQRAGService:
    """Production-ready RAG pipeline with intent routing and optimization."""

    def __init__(self, vectorstore: CliQVectorStore):
        self.vectorstore = vectorstore
        self.classifier_llm = classifier_llm
        self.response_llm = response_llm

        # 1. Contextualize Question
        self.contextualize_chain = self._build_contextualize_chain()

        # 2. Intent Classifier
        self.intent_chain = INTENT_PROMPT | self.classifier_llm | StrOutputParser()

        # 3. Direct response chain for greeting and out-of-scope handling
        self.direct_response_chain = DIRECT_RESPONSE_PROMPT | self.response_llm | StrOutputParser()

        # 4. Document Chain
        self.document_chain = create_stuff_documents_chain(self.response_llm, RAG_PROMPT)
        self.final_answer_chain = RunnableBranch(
            (
                lambda x: x["classification_info"]["response_mode"] == "direct",
                self.direct_response_chain.with_config(tags=["final_response"]),
            ),
            self.document_chain.with_config(tags=["final_response"]),
        )

    def _build_contextualize_chain(self):
        system_prompt = (
            "Given chat history and latest user question, "
            "rewrite it into a standalone question if needed. "
            "Do NOT answer."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        return prompt | self.classifier_llm | StrOutputParser()

    def _get_classification(self, info):
        standalone_question = info["standalone_question"]
        chat_history = info["chat_history"]

        q = standalone_question.lower().strip()

        greetings = {
            "hi",
            "hello",
            "hey",
            "how are you",
            "how r u",
            "who are you",
        }
        if q in greetings:
            intent = "greeting"
            section = "cliq_details"
        else:
            res = self.intent_chain.invoke({
                "input": standalone_question,
                "chat_history": chat_history
            })

            try:
                classification = json.loads(res)
            except Exception:
                classification = {"intent": "out_of_scope", "section": "cliq_details"}

            intent = classification.get("intent", "out_of_scope")
            section = classification.get("section", "cliq_details")

        instructions_map = {
            "greeting": (
                "Reply in 1 short friendly sentence. Welcome the user and offer help with CliQ."
            ),
            "steps": (
                "Answer with Markdown using this format:\n"
                "## Steps\n"
                "- Step 1\n"
                "- Step 2\n"
                "- Step 3\n"
                "Use only steps supported by the provided context. Keep the wording natural and easy to follow."
            ),
            "information": (
                "Answer with Markdown using this format:\n"
                "## Title\n"
                "Short paragraph explaining the feature based only on the context.\n"
                "If useful, add a `### Details` section with short bullet points. Keep the tone conversational and clear."
            ),
            "navigation": (
                "Answer with Markdown using this format:\n"
                "## Navigation\n"
                "State the path clearly, such as `Home > Profile > Settings`, "
                "and then add a short explanation in a natural chat tone."
            ),
            "out_of_scope": (
                "Reply in 1 short polite sentence that you can only help with CliQ platform features, usage, users, posts, messages, requests, and profile-related questions."
            ),
        }

        if intent not in instructions_map:
            intent = "out_of_scope"

        intent_instructions = instructions_map[intent]

        print(f"Flow: Intent='{intent}', Section='{section}'")

        if intent in {"greeting", "out_of_scope"}:
            return {
                "context": [],
                "intent_instructions": intent_instructions,
                "response_mode": "direct",
            }

        pre_filter = {"section": {"$eq": section}}

        docs = self.vectorstore.similarity_search(
            standalone_question,
            k=3,
            pre_filter=pre_filter
        )

        if not docs:
            return {
                "context": [],
                "intent_instructions": instructions_map["out_of_scope"],
                "response_mode": "direct",
            }

        return {
            "context": docs,
            "intent_instructions": intent_instructions,
            "response_mode": "rag",
        }

    def get_chain(self):
        """Main RAG pipeline"""

        return (
            RunnablePassthrough.assign(
                standalone_question=self.contextualize_chain
            )
            | RunnablePassthrough.assign(
                classification_info=RunnableLambda(self._get_classification)
            )
            | RunnablePassthrough.assign(
                context=lambda x: x["classification_info"]["context"],
                intent_instructions=lambda x: x["classification_info"]["intent_instructions"]
            )
            | RunnablePassthrough.assign(
                answer=self.final_answer_chain
            )
        )
