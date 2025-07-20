import re
from internetarchive import download
import PyPDF2
import os
import shutil

DOWNLOADS_DIR = "./downloads"

def ensure_downloads_dir_exists():
    """Ensure the downloads directory exists."""
    if not os.path.exists(DOWNLOADS_DIR):
        os.makedirs(DOWNLOADS_DIR)

def clear_downloads_dir():
    """Delete all contents of the downloads directory."""
    if os.path.exists(DOWNLOADS_DIR):
        for item in os.listdir(DOWNLOADS_DIR):
            item_path = os.path.join(DOWNLOADS_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or symlink
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory

def getArchiveContent(url):
    # Ensure the downloads directory exists
    ensure_downloads_dir_exists()

    # Extract item ID from URL
    item_id = extract_item_id(url)
    
    # Download the PDF
    download(item_id, formats='PDF', glob_pattern='*.pdf', verbose=True, destdir=DOWNLOADS_DIR)
    
    # Read the PDF content
    pdf_content = read_pdf(item_id)
    
    return pdf_content

def extract_item_id(url):
    # Regular expression to extract item ID from URL
    match = re.search(r'/details/([^/]+)', url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Unable to extract item ID from URL")

def read_pdf(item_id):
    pdf_path = os.path.join(DOWNLOADS_DIR, item_id, f"{item_id}.pdf")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    content = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content += page.extract_text()
    
    return content