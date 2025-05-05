from PIL import Image
import os
import shutil

# source - deepseek
def compress_images(input_folder, output_folder, quality=85):
    """
    Try to compress all JPG images in input_folder and save to output_folder.
    If compression fails, copy the original image instead.
    
    Args:
        input_folder: Path to folder containing original images
        output_folder: Path to save compressed/copied images
        quality: Quality setting (1-100), lower means more compression
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate through all files in input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                # Try to open and compress the image
                with Image.open(input_path) as img:
                    img.save(output_path, "JPEG", quality=quality, optimize=True)
                print(f"Successfully compressed: {filename}")
                
            except Exception as e:
                # If compression fails, copy the original
                shutil.copy2(input_path, output_path)
                print(f"Compression failed for {filename}, copied original instead. Error: {str(e)}")

compress_images("../data/images/person", "data/images/person_comp")
compress_images("../data/images/robot", "data/images/robot_comp")
