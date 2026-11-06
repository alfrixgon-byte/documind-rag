import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

class AdvancedRAGPipeline:
    def __init__(self, openai_api_key: str):
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)
        self.vector_store = None
        self.retriever = None

    def ingest_document(self, file_path: str):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="rag_collection"
        )

        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=encoder, top_n=3)

        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )

    def query(self, question: str) -> dict:
        if not self.retriever:
            raise ValueError("No document ingested yet.")

        retrieved_docs = self.retriever.invoke(question)

        prompt_template = """You are an expert AI Assistant. Answer the question based ONLY on the following context. 
If you don't know the answer, say that you don't know. Do not hallucinate.

Context:
{context}

Question: {question}

Answer with page citations:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)

        context_text = "\n\n".join([
            f"[Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}" 
            for doc in retrieved_docs
        ])

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context_text, "question": question})

        return {
            "answer": response,
            "source_documents": retrieved_docs
        }

