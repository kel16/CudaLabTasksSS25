# Code source: ChatGPT
import re
from pathlib import Path
import uuid

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

def rename_images(folder, base_name):
    """ Updates file names to follow count order. """
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"The folder '{folder}' does not exist or is not a directory.")

    # Get all image files, sorted by modification time
    image_files = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ], key=lambda x: x.stat().st_mtime)

    # First pass: rename all to temporary unique names to avoid conflicts
    temp_files = []
    for img in image_files:
        temp_name = f"__tmp__{uuid.uuid4().hex}{img.suffix}"
        temp_path = img.with_name(temp_name)
        img.rename(temp_path)
        temp_files.append(temp_path)

    # Second pass: rename to final normalized names
    for idx, temp_path in enumerate(sorted(temp_files, key=lambda x: x.stat().st_mtime), 1):
        new_name = f"{base_name}_{idx:03d}{temp_path.suffix.lower()}"
        final_path = folder_path / new_name
        print(f"Renaming: {temp_path.name} -> {new_name}")
        temp_path.rename(final_path)

    print(f"\nRenamed {len(temp_files)} images using base name '{base_name}'.")

if __name__=="__main__":
    rename_images("images/person", base_name="person")
    rename_images("images/robot", base_name="robot")
