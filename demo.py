
import os
import pandas as pd
from pathlib import Path
from bert_score import score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

from qanda import QandA


def calculate_bertscore_df(df):
    references = df['CORRECT_ANSWER'].tolist()
    candidates = df['LLM_ANSWER'].tolist()
    
    precision, recall, f1 = score(candidates, references, lang="en", verbose=True)
    
    df['BERT_PRECISION'] = precision.tolist()
    df['BERT_RECALL'] = recall.tolist()
    df['BERT_F1'] = f1.tolist()
    
    return df


FILE_PATH = Path("jsondata/Rodier-Finding.jsonl")
GEN_MODEL = "gemma3"
EMBED_MODEL = "mxbai-embed-large"
VDB = InMemoryVectorStore
TOP_K = 3

PROMPT = ChatPromptTemplate.from_template(
    """Context information is below.
    \n---------------------\n
    {context}
    \n---------------------\n
    Given the context information and not prior knowledge, answer the query.\n
    Query: {input}\n
    Answer:\n""",
)

qanda = QandA(gen_model=GEN_MODEL,
              embed_model=EMBED_MODEL, 
              vdb=VDB,
              file_path=FILE_PATH,
              top_k=TOP_K,
              prompt=PROMPT)


QUESTIONS = ["Who is the coroner?", "Who is the deceased?", "What was the cause of death?"]
CORRECT_ANSWERS = ["Sarah Helen Linton", "Frank Edward Rodier", "unascertained"]
LLM_ANSWERS = []

for i, QUESTION in enumerate(QUESTIONS):
    ANSWER = qanda.ask(QUESTION)
    LLM_ANSWERS.append(ANSWER)
    print(f"Answer {i + 1}: ", ANSWER)

data = {
    'FILENAME': ['Rodier-Finding'] * len(QUESTIONS),
    'MODEL': ['gemma3'] * len(QUESTIONS),
    'QUESTION': QUESTIONS,
    'CORRECT_ANSWER': CORRECT_ANSWERS,
    'LLM_ANSWER': LLM_ANSWERS
}

df = pd.DataFrame(data)
scores_df = calculate_bertscore_df(df)



