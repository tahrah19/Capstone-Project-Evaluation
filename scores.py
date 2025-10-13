
import pandas as pd
import os
import sys
from bert_score import score


def read_data(file_path):
    """
    Reads data from a CSV or XLSX file and returns a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV or XLSX file.
    
    Returns:
        pandas.DataFrame: The data from the file as a DataFrame.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        file_type = 'csv'
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        file_type = 'xlsx'
    else:
        raise ValueError('Unsupported file type. Please provide a CSV or XLSX file.')
    
    return df, file_type


def calculate_bertscore_df(df):
    """
    Calculates the BERTScore for each pair of CORRECT_ANSWER and LLM_ANSWER in the input DataFrame.
    
    Args:
        df (pandas.DataFrame): A DataFrame with required columns 'CORRECT_ANSWER', and 'LLM_ANSWER'.
        
    Returns:
        pandas.DataFrame: A DataFrame with the additional columns 'BERT_PRECISION', 'BERT_RECALL', and 'BERT_F1' containing the BERTScore for each pair.
    """
    df.columns = [c.strip().lower() for c in df.columns]
    references = df['correct_answer'].tolist()
    candidates = df['llm_answer'].tolist()
    
    precision, recall, f1 = score(candidates, references, lang="en", verbose=True)
    
    df['BERT_PRECISION'] = precision.tolist()
    df['BERT_RECALL'] = recall.tolist()
    df['BERT_F1'] = f1.tolist()
    
    return df


if __name__ == "__main__":
    # Ask user for input file
    file_path = input("Enter path to the evaluation file (CSV/XLSX): ").strip().strip('"')

    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        exit(1)

    # Read the data
    data_frame, file_type = read_data(file_path)

    required_cols = ["DOCUMENT", "QUESTION", "CORRECT_ANSWER", "LLM_ANSWER"]
    for col in required_cols:
        if col not in data_frame.columns:
            raise ValueError(f" Missing required column: {col}")

    # Calculate BERTScore
    data_frame = calculate_bertscore_df(data_frame)

    # Save output in same folder as input 
    input_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(input_dir, f"{base_name}-BertScore.xlsx")

    data_frame.to_excel(output_file, index=False)
    print(f"\n BERTScore results saved to:\n{output_file}\n")