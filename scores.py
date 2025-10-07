
import pandas as pd
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
    references = df['CORRECT_ANSWER'].tolist()
    candidates = df['LLM_ANSWER'].tolist()
    
    precision, recall, f1 = score(candidates, references, lang="en", verbose=True)
    
    df['BERT_PRECISION'] = precision.tolist()
    df['BERT_RECALL'] = recall.tolist()
    df['BERT_F1'] = f1.tolist()
    
    return df


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python script.py <file_path>')
        sys.exit(1)
    
    file_path = sys.argv[1]
    file_stem = file_path.rsplit('.', 1)[0]
    data_frame, file_type = read_data(file_path)
    data_frame = calculate_bertscore_df(data_frame)
    
    if file_type == 'csv':
        csv_file_path = file_stem + '-bert-scores.csv'
        data_frame.to_csv(csv_file_path, index=False)
        print(f'CSV file saved to: {csv_file_path}')
    elif file_type == 'xlsx':
        xlsx_file_path = file_stem + '-bert-scores.xlsx'
        data_frame.to_excel(xlsx_file_path, index=False)
        print(f'XLSX file saved to: {xlsx_file_path}')
    else:
        raise ValueError('Unsupported file type. Please provide a CSV or XLSX file.')

