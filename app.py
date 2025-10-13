
import os
import pandas as pd
import streamlit as st
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

from qanda import QandA


from models import get_available_models

available_models = get_available_models()
available_models.remove('mxbai-embed-large')
available_models = [None] + available_models

DATA_DIR = './jsondata'

if os.path.exists(DATA_DIR):
    file_names = os.listdir(DATA_DIR)
else:
    print("No data. Exiting...") # terminal
    st.subheader("No data.")     # app
    exit()

documents = [None] + [os.path.splitext(name)[0] for name in file_names]

# ------------------------------------------------------------------------------ 


#documents = [                     # GET
#    None,                         # RID
#    "Baby-H-finding",             # OF
#    "Blood-results-redacted",     # THIS
#    "Forkin-finding-2014",        # !
#    "Nicholls-Diver-finding",
#    "Rodier-Finding",
#    "TAULELEI-Jacob-Finding"
#]


st.title("Welcome to Coroner App")

st.divider()
st.button("Refresh", on_click=st.cache_resource.clear())

st.subheader("Choose your model")
chosen_model = st.selectbox("", available_models)

st.subheader("Choose your document")
chosen_document = st.selectbox("", documents)


if chosen_document != None:

    FILE_PATH = Path('jsondata/' + chosen_document + '.jsonl')
    #FILE_PATH = Path("DoesNotExist")

    if os.path.exists(FILE_PATH):
        st.write(FILE_PATH)
    else:
        st.write("Error: no data")

    #GEN_MODEL = "gemma3"        # GET RID OF THIS!
    GEN_MODEL = chosen_model
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

    @st.cache_resource
    def qanda():
        return QandA(gen_model=GEN_MODEL,
                     embed_model=EMBED_MODEL, 
                     vdb=VDB,
                     file_path=FILE_PATH,
                     top_k=TOP_K,
                     prompt=PROMPT)

    qanda = qanda()

    st.text("")
    st.text("")
    st.subheader("Ask a question")
    question = st.text_area("", None)

    if question != None:
        #answer = qanda.ask(str(question))
        answer, sources = qanda.ask(str(question), verbose=True)
        st.write(answer)
        st.write(sources)

    # --------------------------------------------------------------------------
    # Evaluation Section (Automatic QA Evaluation)
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("Automatic Evaluation")

    eval_file = st.file_uploader("📂 Upload a question file (CSV or XLSX)", type=["csv", "xlsx"])

    if eval_file is not None:
        df = pd.read_excel(eval_file) if eval_file.name.endswith('.xlsx') else pd.read_csv(eval_file)
        df.columns = [c.strip().upper() for c in df.columns]

        required_cols = ['QUESTION', 'CORRECT_ANSWER']
        if not all(col in df.columns for col in required_cols):
            st.error(f" Missing required columns. Expected: {required_cols}")
        else:
            st.success(" File loaded successfully! Starting evaluation...")

            model_answers = []
            progress = st.progress(0)

            for i, row in df.iterrows():
                question = str(row['QUESTION']).strip()
                st.write(f"**Question {i+1}/{len(df)}:** {question}")

                try:
                    retrieved_docs = qanda.retriever.invoke(question)
                    context = "\n\n".join([d.page_content for d in retrieved_docs])

                    response = qanda.llm.invoke(
                        qanda.prompt.format(context=context, input=question)
                    )

                    model_answer = response.strip() if isinstance(response, str) else getattr(response, "content", str(response)).strip()
                except Exception as e:
                    model_answer = f"Error: {e}"

                model_answers.append(model_answer)
                progress.progress((i + 1) / len(df))

            df['LLM_ANSWER'] = model_answers
            df['DOCUMENT'] = chosen_document
            df = df[['DOCUMENT', 'QUESTION', 'CORRECT_ANSWER', 'LLM_ANSWER']]

            # Save output in evaluations folder
            os.makedirs("evaluations", exist_ok=True)
            output_filename = f"{os.path.splitext(eval_file.name)[0]}-{GEN_MODEL}.csv"
            output_path = os.path.join("evaluations", output_filename)
            df.to_csv(output_path, index=False)

            st.success(f"Evaluation complete! Results saved to `{output_path}`")
            st.dataframe(df)

            # --- Download button ---
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Evaluation Results",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv"
            )
