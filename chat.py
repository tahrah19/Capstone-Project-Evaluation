
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

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from langchain_core.vectorstores import InMemoryVectorStore

from vecdb import VectorStore
from models import ollama_status, select_model
from utils import choose_file, clip_text
from converters import pdf_converter, extract_metadata, metadata


def initialize_from_pdf(gen_model,
                        embed_model,
                        vdb,
                        file_path,
                        ocr, 
                        converter, 
                        export_type,
                        max_tokens, 
                        top_k):

    print("\nInitializing, please wait...\n")

    llm = OllamaLLM(model=gen_model)
    embedding = OllamaEmbeddings(model=embed_model)

    if os.path.exists(file_path):
        print(f"Loading {file_path}")
        loader = DoclingLoader(
            converter=converter,
            file_path=file_path,
            export_type=export_type
        )
    else:
        print(f"Error loading file.\n\nExiting...")
        exit()
        
    loaded_documents = loader.load()
    documents = extract_metadata(loaded_documents)

    vectorstore = VectorStore(vdb=vdb, embedding=embedding, documents=documents)
    retriever = vectorstore.retriever(top_k=top_k)

    print("\nLLM & vector store ready.\n")

    return llm, retriever


def initialize_from_json(gen_model,
                         embed_model,
                         vdb,
                         file_path,
                         top_k):

    print("\nInitializing, please wait...\n")

    llm = OllamaLLM(model=gen_model)
    embedding = OllamaEmbeddings(model=embed_model)

    if os.path.exists(file_path):
        print(f"Loading {file_path}")
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.',
            content_key='page_content',
            metadata_func=metadata,
            json_lines=True
        )
    else:
        print(f"Error loading file.\n\nExiting...")
        exit()

    documents = loaded_documents = loader.load()

    vectorstore = VectorStore(vdb=vdb, embedding=embedding, documents=documents)
    retriever = vectorstore.retriever(top_k=top_k)

    print("\nLLM & vector store ready.\n")

    return llm, retriever


def interactive_chat(llm, retriever, prompt):

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    while True:

        print('\n\n' + 80 * '-' + '\n\n')
        question = input("Ask your question (type q to quit): ")
        print('\n')

        if question == 'q':
            break

        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": question})

        answer = response["answer"]
        print(f"Question:\n{response['input']}\n\nAnswer:\n{answer}")

        for i, doc in enumerate(response["context"]):
            print(f"\nSource {i + 1}:")
            print(f"  * text: {json.dumps(clip_text(doc.page_content, threshold=350))}")
            print(f"  * page: {doc.metadata.get('page_no')}")
            print(f"  * document: {doc.metadata.get('source')}")
import pandas as pd
from tqdm import tqdm
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def evaluate_questions(llm, retriever, prompt, question_file, output_file):
    """
    Automatic evaluation mode.
    Reads questions and reference answers from a CSV/XLSX file, queries the LLM,
    and saves model answers in the same structure.
    """

    print("\nStarting automatic evaluation...\n")

    # Load the question file (accepts CSV or XLSX)
    if question_file.endswith(".xlsx"):
        df = pd.read_excel(question_file)
    else:
        df = pd.read_csv(question_file)

    # Expected columns
    required_cols = ["document", "question", "correct_answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Missing column '{col}'.\nExpected columns: {required_cols}\n"
                f"Please use the same format as bert_input_template.csv."
            )

    # Prepare the retrieval + question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating questions"):
        question = row["question"]
        document = row["document"]
        correct_answer = row["correct_answer"]

        try:
            response = rag_chain.invoke({"input": question})
            llm_answer = response.get("answer", "").strip()
        except Exception as e:
            llm_answer = f"Error: {e}"

        results.append({
            "document": document,
            "question": question,
            "correct_answer": correct_answer,
            "llm_answer": llm_answer
        })

    out_df = pd.DataFrame(results)

    # Save output in the same lowercase format
    if output_file.endswith(".xlsx"):
        out_df.to_excel(output_file, index=False)
    else:
        out_df.to_csv(output_file, index=False)

    print(f"\nEvaluation completed — results saved to: {output_file}\n")


if __name__ == "__main__":

    if ollama_status() == False:
        print("Exiting...\n")
        exit()

    GEN_MODEL = select_model()
    
    EMBED_MODEL = "mxbai-embed-large"
    VDB = InMemoryVectorStore
    TOP_K = 3

    PDF_DATA = Path("./data")
    JSON_DATA = Path("./jsondata")
    #JSON_DATA = Path("./DoesNotExist")

    if os.path.exists(JSON_DATA):
        DATA_DIR = JSON_DATA
    elif os.path.exists(PDF_DATA):
        DATA_DIR = PDF_DATA
    else:
        print("Error: no data\n\nExiting...\n")
        exit()

    FILE_PATH = choose_file(DATA_DIR)

    PROMPT = ChatPromptTemplate.from_template(
        """Context information is below.
        \n---------------------\n
        {context}
        \n---------------------\n
        Given the context information and not prior knowledge, answer the query.\n
        Query: {input}\n
        Answer:\n""",
    )


    if DATA_DIR == JSON_DATA:

        LLM, RETRIEVER = initialize_from_json(gen_model=GEN_MODEL,
                                              embed_model=EMBED_MODEL,
                                              vdb=VDB,
                                              file_path=FILE_PATH,
                                              top_k=TOP_K)
    elif DATA_DIR == PDF_DATA:

        OCR = True if os.path.basename(FILE_PATH) == "Nicholls-Diver finding.pdf" else False
        CONVERTER = pdf_converter(OCR=OCR)

        EXPORT_TYPE = ExportType.DOC_CHUNKS
        MAX_TOKENS = 512

        LLM, RETRIEVER = initialize_from_pdf(gen_model=GEN_MODEL,
                                             embed_model=EMBED_MODEL,
                                             vdb=VDB,
                                             file_path=FILE_PATH,
                                             ocr=OCR,
                                             converter=CONVERTER,
                                             export_type=EXPORT_TYPE,
                                             max_tokens=MAX_TOKENS,
                                             top_k=TOP_K)

    print("Starting chat.\n")

    # -------------------------------
    # MODE SELECTION (interactive vs auto-run)
    # -------------------------------

    print("\nSelect mode:")
    print("1 - Interactive (manual Q&A)")
    print("2 - Auto-run (read from question file)\n")

    mode = input("Enter mode [1 or 2]: ").strip()

    if mode == "2":
        question_file = input("Enter path to question file (CSV/XLSX): ").strip()
        output_file = input("Enter desired output filename (e.g., file-name-llama3.2.xlsx): ").strip()
        evaluate_questions(LLM, RETRIEVER, PROMPT, question_file, output_file)
    else:
        interactive_chat(llm=LLM, retriever=RETRIEVER, prompt=PROMPT)