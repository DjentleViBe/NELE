# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Operations involving files
"""
import os
import shutil
import re

def reset_directory(path):
    """
    If the directory exists, delete all its contents.
    If it doesn't exist, create it.
    """
    if os.path.exists(path):
        # Remove all contents inside the directory
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    else:
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

def create_directory(path):
    """
    Create a new directory
    """
    os.makedirs(path, exist_ok=True)

def getlatest(folder):
    """
    Docstring for getlatest
    
    :param folder: Folder path
    """
    files = [f for f in os.listdir(folder) if f.endswith(".pth")]
    # Sort by number
    files_sorted = sorted(files, key=extract_number, reverse=True)

    # Get the latest file
    latest_file = files_sorted[0] if files_sorted else None

    return folder + latest_file

def extract_number(filename):
    """
    Extract number from filename
    
    :param filename: location of filenames
    """
    match = re.search(r"_(\d+)\.pth$", filename)
    return int(match.group(1)) if match else -1
