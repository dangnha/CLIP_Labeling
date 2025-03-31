import os
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
from io import BytesIO
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, upload_folder):
    if file.filename == '':
        raise ValueError("No file selected")
    
    if not allowed_file(file.filename):
        raise ValueError("File type not allowed")
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large. Max size is {MAX_FILE_SIZE//(1024*1024)}MB")
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = secure_filename(file.filename)
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{timestamp}{ext}"
    filepath = os.path.join(upload_folder, unique_filename)
    
    # Ensure directory exists
    os.makedirs(upload_folder, exist_ok=True)
    
    # Validate and save image
    try:
        img = Image.open(BytesIO(file.read()))
        img.verify()
        file.seek(0)
        file.save(filepath)
        return filepath, unique_filename
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

def delete_file(filepath):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False
    except Exception as e:
        raise Exception(f"Error deleting file: {str(e)}")