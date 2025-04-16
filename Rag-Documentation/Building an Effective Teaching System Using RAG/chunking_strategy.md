# Improved Chunking Strategy for Math Educational Content

## Analysis of Current System

The current RAG system uses a basic chunking approach with `RecursiveCharacterTextSplitter` with a chunk size of 5000 characters and a chunk overlap of 500 characters. While this approach works for general text, it doesn't account for the specific structure of educational content, particularly math instruction that simulates classroom teaching.

## Proposed Chunking Strategy

Based on the analysis of the educational content files, I propose a semantic chunking strategy that preserves the pedagogical structure of math instruction. This approach will ensure that related concepts stay together and the teaching flow remains coherent.

### 1. Hierarchical Chunking Structure

Instead of using a fixed character count, we'll chunk the content based on its semantic structure:

#### Level 1: Topic-Level Chunks
- Each math topic (Algebra, Geometry, etc.) will have a main chunk containing:
  - Topic overview
  - Prerequisites
  - Learning objectives
  - Why this topic is important

#### Level 2: Concept-Level Chunks
- Each core concept within a topic will be its own chunk:
  - Concept definition and explanation
  - Key formulas and properties
  - Teaching methodology for this concept
  - Basic examples

#### Level 3: Application-Level Chunks
- Practice problems and applications for each concept:
  - Step-by-step solutions
  - Common mistakes and misconceptions
  - Real-world applications
  - Advanced examples

#### Level 4: Meta-Information Chunks
- Teaching strategies specific to the concept
- Transition guidance between topics
- Assessment strategies
- Differentiation approaches for various learning levels

### 2. Metadata Enhancement

Each chunk will include metadata to improve retrieval relevance:

```json
{
  "topic": "Algebra",
  "concept": "Quadratic Equations",
  "difficulty_level": "Intermediate",
  "prerequisites": ["Linear Equations", "Basic Algebraic Operations"],
  "teaching_stage": "Concept Introduction",
  "content_type": "Explanation"
}
```

### 3. Cross-Reference System

To maintain connections between related chunks:

- Each chunk will contain references to prerequisite chunks
- Each chunk will contain references to follow-up chunks
- Concept chunks will reference related practice problem chunks

### 4. Implementation Approach

We'll implement this chunking strategy using a combination of:

1. **Custom Text Splitter**: Create a custom splitter that recognizes markdown headers and section boundaries

```python
class MathEducationSplitter:
    def split_documents(self, documents):
        chunks = []
        for doc in documents:
            # Split by main headers (Topic level)
            topic_sections = self._split_by_pattern(doc.page_content, r'# [^\n]+\n')
            
            for topic_section in topic_sections:
                # Extract topic metadata
                topic_name = self._extract_topic_name(topic_section)
                
                # Split by concept headers
                concept_sections = self._split_by_pattern(topic_section, r'## [^\n]+\n')
                
                for concept_section in concept_sections:
                    # Extract concept metadata
                    concept_name = self._extract_concept_name(concept_section)
                    
                    # Create concept chunk with metadata
                    chunks.append(Document(
                        page_content=concept_section,
                        metadata={
                            "topic": topic_name,
                            "concept": concept_name,
                            "source": doc.metadata.get("source"),
                            "chunk_type": "concept"
                        }
                    ))
                    
                    # Additional processing for practice problems, etc.
                    
        return chunks
```

2. **Metadata Extraction**: Extract relevant metadata from each section based on content patterns

3. **Chunk Overlap Strategy**: Ensure that key information (like prerequisites) appears in multiple relevant chunks

### 5. Chunk Size Considerations

Rather than using a fixed character count:

- **Topic Overview Chunks**: ~1000-1500 characters
- **Concept Explanation Chunks**: ~2000-3000 characters
- **Practice Problem Chunks**: ~1500-2500 characters per problem set
- **Teaching Strategy Chunks**: ~1000-2000 characters

This variable sizing ensures that pedagogically complete units stay together.

## Benefits of This Approach

1. **Pedagogical Coherence**: Chunks align with how teachers actually present material
2. **Improved Retrieval Relevance**: Metadata helps retrieve the most appropriate teaching content
3. **Context Preservation**: Related concepts stay together
4. **Teaching Flow Maintenance**: The system can follow a natural teaching progression
5. **Prerequisite Awareness**: The system can check and address prerequisite knowledge gaps

## Implementation Code Example

```python
from langchain.text_splitters import RecursiveCharacterTextSplitter
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
            overview_match = re.search(r'#+\s+.*?Overview.*?\n(.*?)(?=#+\s+)', content, re.DOTALL)
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
            prereq_match = re.search(r'#+\s+.*?Prerequisite.*?\n(.*?)(?=#+\s+)', content, re.DOTALL)
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
```

## Integration with Existing System

This chunking strategy will replace the current `RecursiveCharacterTextSplitter` in the app.py file. The implementation will require:

1. Creating the custom chunker class
2. Modifying the document loading and processing pipeline
3. Updating the vector store creation to include the enhanced metadata
4. Adjusting the retrieval mechanism to leverage the metadata for more relevant results

By implementing this improved chunking strategy, the RAG system will better simulate how a teacher presents math concepts in a classroom, providing a more coherent and effective learning experience for students.
