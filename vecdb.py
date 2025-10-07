
class VectorStore:
    """
    A class that manages a vector database (vdb) for storing and retrieving document embeddings.

    Attributes:
        vdb (VectorDatabase): The vector database instance used for storing and querying document embeddings.
        embedding (Embedding): The embedding model used to generate document embeddings.
        documents (List[Document]): The list of documents to be stored in the vector database.
        vectorstore (VectorStore): The configured vector store instance for storing and retrieving document embeddings.

    Methods:
        retriever(top_k: int) -> Retriever:
            Returns a retriever instance that can be used to query the vector database and retrieve the top-k most relevant documents.
    """
    def __init__(self, vdb, embedding, documents):
        self.vdb = vdb
        self.embedding = embedding
        self.documents = documents
        self.vectorstore = vdb.from_documents(
            documents=self.documents,
            embedding=self.embedding,
        )

    def retriever(self, top_k):
        """
        Returns a retriever instance that can be used to query the vector database and retrieve the top-k most relevant documents.

        Args:
            top_k (int): The number of most relevant documents to retrieve.

        Returns:
            Retriever: The configured retriever instance.
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})


