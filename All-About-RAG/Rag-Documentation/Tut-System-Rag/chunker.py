from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re

class MathTeacherChunker:
    def __init__(self):
        # Base splitter for fallback
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
    
    def split_documents(self, documents):
        all_chunks = []
        
        for doc in documents:
            # Extract filename to determine topic
            filename = doc.metadata.get("source", "").split("/")[-1]
            topic = filename.split(".")[0] if "." in filename else "Unknown"
            
            content = doc.page_content
            
            # Extract topic overview
            overview_match = re.search(r'#+\s+.*?Overview.*?\n(.*?)(?=#+\s+|\Z)', content, re.DOTALL)
            if overview_match:
                overview_chunk = Document(
                    page_content=overview_match.group(0),
                    metadata={
                        "topic": topic,
                        "chunk_type": "overview",
                        "source": doc.metadata.get("source")
                    }
                )
                all_chunks.append(overview_chunk)
            
            # Extract prerequisites
            prereq_match = re.search(r'#+\s+.*?Prerequisite.*?\n(.*?)(?=#+\s+|\Z)', content, re.DOTALL)
            if prereq_match:
                prereq_chunk = Document(
                    page_content=prereq_match.group(0),
                    metadata={
                        "topic": topic,
                        "chunk_type": "prerequisites",
                        "source": doc.metadata.get("source")
                    }
                )
                all_chunks.append(prereq_chunk)
            
            # Extract concepts (using level 2 headers)
            concept_matches = re.finditer(r'(##\s+.*?(?=##|\Z))', content, re.DOTALL)
            for match in concept_matches:
                concept_content = match.group(1)
                concept_title_match = re.search(r'##\s+(.*?)\n', concept_content)
                concept_name = concept_title_match.group(1) if concept_title_match else "Unknown Concept"
                
                concept_chunk = Document(
                    page_content=concept_content,
                    metadata={
                        "topic": topic,
                        "concept": concept_name,
                        "chunk_type": "concept",
                        "source": doc.metadata.get("source")
                    }
                )
                all_chunks.append(concept_chunk)
                
                # Extract practice problems related to this concept
                problem_match = re.search(r'(?:Example|Problem|Practice).*?\n(.*?)(?=##|\Z)', concept_content, re.DOTALL)
                if problem_match:
                    problem_chunk = Document(
                        page_content=problem_match.group(0),
                        metadata={
                            "topic": topic,
                            "concept": concept_name,
                            "chunk_type": "practice",
                            "source": doc.metadata.get("source")
                        }
                    )
                    all_chunks.append(problem_chunk)
            
            # Extract teaching strategies
            teaching_match = re.search(r'#+\s+.*?Teaching.*?\n(.*?)(?=#+\s+|\Z)', content, re.DOTALL)
            if teaching_match:
                teaching_chunk = Document(
                    page_content=teaching_match.group(0),
                    metadata={
                        "topic": topic,
                        "chunk_type": "teaching_strategy",
                        "source": doc.metadata.get("source")
                    }
                )
                all_chunks.append(teaching_chunk)
        
        # If no chunks were created, fall back to base splitter
        if not all_chunks:
            return self.base_splitter.split_documents(documents)
            
        return all_chunks
