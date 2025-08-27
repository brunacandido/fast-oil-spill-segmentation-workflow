import os

def get_unique_path(path):
    """
    If file already exists create a new one.
    file.txt -> file_1.txt -> file_2.txt
    """
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path

    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1

    return new_path
