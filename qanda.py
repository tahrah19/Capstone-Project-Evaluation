
import os
import json
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import JSONLoader
from langchain_core.vectorstores import InMemoryVectorStore
from vecdb import VectorStore
from converters import metadata


class QandA:
    """
    A class for performing question-answering tasks using a language model and a vector database.

    Attributes:
        gen_model (str): The name of the language model to be used for generating answers.
        embed_model (str): The name of the embedding model to be used for generating embeddings.
        vdb (str): The name of the vector database to be used for storing and retrieving documents.
        file_path (str): The path to the file containing the documents to be used for the question-answering task.
        top_k (int): The number of top-k most relevant documents to be retrieved for each question.
        prompt (str): The prompt to be used for the question-answering chain.

    Methods:
        ask(question, verbose=False):
            Invokes the question-answering chain to generate an answer to the given question.
            Args:
                question (str): The question to be answered.
                verbose (bool, optional): If True, the method will return the answer and the sources used to generate the answer. Defaults to False.
            Returns:
                str or (str, list): The answer to the question, or a tuple containing the answer and the sources used to generate the answer.
    """
    def __init__(self, gen_model, embed_model, vdb, file_path, top_k, prompt):
        self.gen_model = gen_model
        self.embed_model = embed_model
        self.vdb = vdb
        self.file_path = file_path
        self.top_k = top_k
        self.prompt = prompt

        print("Initializing, please wait...")

        self.llm = OllamaLLM(model=self.gen_model)
        self.embedding = OllamaEmbeddings(model=self.embed_model)

        if os.path.exists(self.file_path):
            print(f"Loading {file_path}")
            loader = JSONLoader(
                file_path=self.file_path,
                jq_schema='.',
                content_key='page_content',
                metadata_func=metadata,
                json_lines=True
            )
        else:
            print(f"Error loading file.\nExiting...")
            return None

        self.documents = loaded_documents = loader.load()
        vectorstore = VectorStore(vdb=self.vdb, embedding=self.embedding, documents=self.documents)
        self.retriever = vectorstore.retriever(top_k=self.top_k)
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)

        print("Question Answer chain ready.")

    def ask(self, question, verbose=False):
        response = self.rag_chain.invoke({"input": question})
        answer = response["answer"]
        if not verbose:
            return answer
        else:
            answer = response["answer"]
            sources = []
            for i, doc in enumerate(response["context"]):
                text = doc.page_content
                page = doc.metadata.get("page_no")
                document = doc.metadata.get("source")
                source = {"source": i+1, "text": text, "page": page, "document": document}
                sources.append(source)
            return answer, sources


