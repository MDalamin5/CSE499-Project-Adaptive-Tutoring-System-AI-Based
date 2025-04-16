import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document

def load_math_content(data_dir):
    """
    Load math content from markdown files in the data directory.
    
    Args:
        data_dir (str): Path to the data directory.
        
    Returns:
        list: List of Document objects.
    """
    documents = []
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Create topic subdirectories if they don't exist
    topics = ["algebra", "geometry", "trigonometry", "functions"]
    for topic in topics:
        os.makedirs(os.path.join(data_dir, topic), exist_ok=True)
    
    # Copy sample content files to appropriate directories if they don't exist yet
    sample_files = {
        "Algebra.md": os.path.join(data_dir, "algebra", "algebra.md"),
        "Geometry.md": os.path.join(data_dir, "geometry", "geometry.md"),
        "Trigonomity.md": os.path.join(data_dir, "trigonometry", "trigonometry.md"),
        "functin-graph.md": os.path.join(data_dir, "functions", "functions.md")
    }
    
    for source, dest in sample_files.items():
        if os.path.exists(f"/home/ubuntu/upload/{source}") and not os.path.exists(dest):
            with open(f"/home/ubuntu/upload/{source}", "r") as src_file:
                content = src_file.read()
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w") as dst_file:
                    dst_file.write(content)
    
    # Load all markdown files from the data directory
    for md_file in glob.glob(os.path.join(data_dir, "**/*.md"), recursive=True):
        try:
            loader = TextLoader(md_file)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {md_file}: {e}")
    
    return documents

def extract_topic_from_filename(filename):
    """
    Extract the topic from a filename.
    
    Args:
        filename (str): The filename.
        
    Returns:
        str: The extracted topic.
    """
    basename = os.path.basename(filename).lower()
    
    if "algebra" in basename:
        return "algebra"
    elif "geometry" in basename:
        return "geometry"
    elif "trigon" in basename:
        return "trigonometry"
    elif "function" in basename or "graph" in basename:
        return "functions"
    
    return "general"
