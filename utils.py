
import os
import re


def format_filename(filename):
    """
    Format a filename by removing whitespace and punctuation, and keeping the 
    file extension.
    
    Args:
        filename (str): The input filename to be formatted.
    
    Returns:
        str: The formatted filename.
    """
    # Remove whitespace and punctuation, except for the file extension
    base, ext = os.path.splitext(filename)
    formatted_base = re.sub(r'[\s\W_]+', '-', base)
    
    # Combine the formatted base and the file extension
    formatted_filename = f"{formatted_base}{ext}"
    
    return formatted_filename


def format_files(directory, verbose=False):
    """
    Formats all the filenames in the specified directory.
    
    Args:
        directory (str): The path to the directory.
    
    Returns:
        None
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            formatted_filename = format_filename(filename)
            new_file_path = os.path.join(directory, formatted_filename)
            os.rename(file_path, new_file_path)
            if verbose:
                print(f"Renamed '{filename}' to '{formatted_filename}'")


def check_file_stem_exists(file_name, directory):
    """
    Checks if the stem of a file name exists in the specified directory.

    Args:
        file_name (str): The name of the file to check.
        directory (str): The directory to search in.

    Returns:
        bool: True if the stem of the file name exists in the directory, 
              False otherwise.
    """
    # Get the stem of the file name (the part before the extension)
    stem, _ = os.path.splitext(file_name)

    # Check if the stem exists as a file in the directory
    for filename in os.listdir(directory):
        if os.path.splitext(filename) == stem:
            return True

    return False


def choose_file(data_dir):
    """
    Displays the files in the specified data directory and prompts the user to choose a file.
    
    Args:
        data_dir (str): The path to the data directory.
    
    Returns:
        str: The path to the selected file.
    """
    # Get a list of files in the data directory
    files = os.listdir(data_dir)
    
    # Display the files to the user
    print("\nAvailable files:\n")
    for i, file in enumerate(files, start=1):
        file_path, _ = os.path.splitext(file)
        file_name = os.path.basename(file_path)
        #print(f"  {i}. {file_name}")
        print(f"  {i}. {file}")
    
    # Prompt the user to choose a file
    while True:
        try:
            choice = int(input("\nEnter the number of the file you want to use: "))
            if 1 <= choice <= len(files):
                return os.path.join(data_dir, files[choice - 1])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def clip_text(text, threshold=256):
    """
    Clips the given text to a maximum length if it exceeds the specified threshold.

    Args:
        text (str): The input text to be clipped.
        threshold (int, optional): The maximum length of the text.

    Returns:
        str: The clipped text with an ellipsis (...) appended if the original text exceeded the threshold.
    """
    return f"{text[:threshold]}..." if len(text) > threshold else text


