import os 
from pathlib import Path

PROJECT_ROOT =  os.path.abspath(os.path.join(__file__, "..", "..", ".."))

def get_path(name, search_dir):
    search_dir = Path(search_dir)
    for folder in search_dir.iterdir():
        if folder.is_dir() and folder.name == name:
            return str(folder.resolve())  # Returns absolute path
    raise FileNotFoundError(f"Folder '{name}' not found in '{search_dir}'")
