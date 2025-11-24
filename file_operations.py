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
    os.makedirs(path, exist_ok=True)

def getlatest(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".pth")]
    # Sort by number
    files_sorted = sorted(files, key=extract_number, reverse=True)

    # Get the latest file
    latest_file = files_sorted[0] if files_sorted else None

    return folder + latest_file

# Extract number from filename
def extract_number(filename):
    match = re.search(r"_(\d+)\.pth$", filename)
    return int(match.group(1)) if match else -1
