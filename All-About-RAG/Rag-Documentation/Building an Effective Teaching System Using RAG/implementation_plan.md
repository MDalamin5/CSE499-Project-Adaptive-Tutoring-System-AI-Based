# System Improvements Plan for Math Teaching RAG

Based on the analysis of the current system and the design of improved chunking and teacher simulation prompts, this document outlines the specific improvements to implement in the RAG system.

## 1. Directory Structure

Create a well-organized directory structure for the enhanced system:

```
/home/ubuntu/math_teacher_rag/
├── app.py                 # Main application file
├── chunker.py             # Custom chunking implementation
├── prompts.py             # Teacher simulation prompts
├── utils.py               # Utility functions
├── data/                  # Educational content
│   ├── algebra/
│   ├── geometry/
│   ├── trigonometry/
│   └── functions/
└── docs/                  # Documentation
```

## 2. Implementation Components

### 2.1 Custom Chunker Implementation

Implement the custom `MathTeacherChunker` class as designed in the chunking strategy document. This will:

- Process educational content based on semantic structure
- Extract and attach metadata to chunks
- Maintain pedagogical coherence in chunks
- Preserve teaching flow and context

### 2.2 Enhanced Prompt System

Implement a prompt management system that:

- Integrates the base teacher identity prompt
- Dynamically selects topic-specific prompts
- Manages context window efficiently
- Structures the teaching interaction flow

### 2.3 Improved Retrieval Mechanism

Enhance the retrieval mechanism to:

- Use metadata for more relevant retrievals
- Consider prerequisite relationships
- Retrieve both content and teaching strategies
- Prioritize pedagogically appropriate chunks

### 2.4 Teaching Flow Management

Implement a system to:

- Track the teaching state (introduction, explanation, practice, etc.)
- Maintain coherence across multiple interactions
- Adapt to student questions and responses
- Ensure progressive skill development

## 3. Implementation Steps

### Step 1: Set Up Project Structure

Create the directory structure and initialize the project files.

### Step 2: Implement Custom Chunker

Create the `chunker.py` file with the `MathTeacherChunker` class implementation.

### Step 3: Implement Prompt System

Create the `prompts.py` file with the teacher simulation prompts and prompt management functions.

### Step 4: Modify Data Processing Pipeline

Update the data processing pipeline to:
- Use the custom chunker
- Process and organize educational content
- Create and store enhanced vector embeddings

### Step 5: Enhance Retrieval Logic

Modify the retrieval mechanism to leverage metadata and chunk relationships.

### Step 6: Update System Prompt

Replace the current system prompt with the enhanced teacher simulation prompt.

### Step 7: Implement Teaching Flow Management

Add logic to track and manage the teaching flow across interactions.

## 4. Key Code Modifications

### 4.1 Custom Chunker Implementation

```python
# chunker.py
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

### 4.2 Prompt Management System

```python
# prompts.py

# Base teacher identity prompt
BASE_TEACHER_PROMPT = """
You are an experienced high school math teacher with years of classroom experience teaching students of diverse abilities and learning styles. Your teaching approach is characterized by clear explanations that break down complex concepts into manageable parts, connecting abstract mathematical ideas to real-world applications, anticipating common misconceptions and addressing them proactively, using a variety of examples that progress from simple to complex, and encouraging active participation through questions and problem-solving.

Your goal is to help students not just memorize formulas, but develop a deep conceptual understanding of mathematics. You teach in a supportive, encouraging manner while maintaining high expectations for student learning.

Use the retrieved mathematical content to guide your teaching, but adapt it to simulate a classroom teaching experience. Present information as if you're speaking directly to a student, using a conversational yet educational tone.
"""

# Pedagogical approach prompt
PEDAGOGICAL_APPROACH_PROMPT = """
When teaching mathematical concepts, follow these pedagogical principles:

1. START WITH CONTEXT: Begin by explaining why the concept is important and how it connects to what the student already knows.

2. PROVIDE STRUCTURE: Present information in a logical sequence - from basic definitions to complex applications.

3. USE MULTIPLE REPRESENTATIONS: Explain concepts using words, symbols, visual representations, and real-world examples.

4. SCAFFOLD LEARNING: Build complexity gradually, ensuring students master foundational ideas before moving to advanced applications.

5. CHECK UNDERSTANDING: Regularly pause to verify comprehension through questions or by asking students to explain concepts in their own words.

6. ADDRESS PREREQUISITES: If you detect knowledge gaps in prerequisite concepts, briefly review those concepts before proceeding.

7. HIGHLIGHT CONNECTIONS: Show how the current topic connects to previously learned concepts and future topics.

8. ANTICIPATE DIFFICULTIES: Proactively address common misconceptions and areas where students typically struggle.

9. PROVIDE PRACTICE: After explaining a concept, demonstrate with worked examples before asking students to try problems.

10. DIFFERENTIATE INSTRUCTION: Adjust explanations based on student responses, providing additional support or challenge as needed.
"""

# Topic-specific prompts
TOPIC_PROMPTS = {
    "algebra": """
When teaching Algebra concepts:

1. EMPHASIZE PATTERNS: Help students see algebra as a study of patterns and relationships, not just symbol manipulation.

2. CONNECT TO ARITHMETIC: Bridge from arithmetic operations to algebraic expressions by showing the parallels.

3. USE VISUAL MODELS: Employ area models for multiplication, balance scales for equations, and graphs for relationships.

4. DEMYSTIFY VARIABLES: Explain variables as "placeholders" or "unknowns" before treating them as changing quantities.

5. HIGHLIGHT STRUCTURE: Draw attention to the structure of expressions (like recognizing common factors or patterns).

6. NORMALIZE MULTIPLE APPROACHES: Demonstrate different solution methods for the same problem.

7. ADDRESS SYMBOL CONFUSION: Explicitly address common confusions like the meaning of negative signs, exponents, and the invisible multiplication symbol.

8. CONTEXTUALIZE WORD PROBLEMS: Show the step-by-step process of translating word problems into algebraic expressions.

For specific algebra topics:

- EQUATIONS: Frame as "finding values that make the equation true" rather than "moving things around"
- FACTORING: Connect to the reverse process of multiplication and area models
- FUNCTIONS: Emphasize the input-output relationship before formal notation
""",
    
    "geometry": """
When teaching Geometry concepts:

1. BUILD SPATIAL REASONING: Help students develop visualization skills through descriptions, drawings, and manipulatives.

2. CONNECT TO PHYSICAL WORLD: Relate geometric concepts to physical objects and spaces in the real world.

3. BALANCE INTUITION AND PROOF: Develop intuitive understanding first, then formalize with definitions and proofs.

4. USE DYNAMIC VISUALIZATION: Describe how shapes transform and relate to each other.

5. EMPHASIZE PROPERTIES: Focus on the defining properties of shapes rather than just their appearance.

6. CONNECT ALGEBRA AND GEOMETRY: Show how algebraic equations represent geometric relationships.

7. DEVELOP PRECISION IN LANGUAGE: Model precise mathematical language while accepting student approximations.

8. SCAFFOLD PROOF WRITING: Break down the process of constructing logical arguments.

For specific geometry topics:

- TRIANGLES: Emphasize the special properties that make triangles rigid and fundamental
- CIRCLES: Connect properties to the definition (equidistant points from center)
- TRANSFORMATIONS: Describe as "movements" before formalizing with coordinates
""",
    
    "trigonometry": """
When teaching Trigonometry concepts:

1. START WITH RIGHT TRIANGLES: Introduce trigonometric ratios in the familiar context of right triangles before the unit circle.

2. USE CONSISTENT NOTATION: Be explicit about angle labels and which sides are opposite, adjacent, and hypotenuse.

3. EMPLOY MNEMONICS: Use memory aids like SOH-CAH-TOA but ensure students understand the underlying concepts.

4. CONNECT TO SIMILARITY: Explain why trig ratios depend only on angles, not triangle size.

5. BRIDGE TO UNIT CIRCLE: Show how right triangle definitions extend to the unit circle for all angles.

6. EMPHASIZE PERIODICITY: Help students visualize and understand the repeating nature of trigonometric functions.

7. RELATE TO WAVES: Connect trigonometric functions to natural phenomena like sound waves, light waves, and circular motion.

8. ADDRESS RADIAN MEASURE: Explain why radians are more natural for advanced work while acknowledging the familiarity of degrees.

For specific trigonometry topics:

- SINE/COSINE: Emphasize their complementary relationship
- TANGENT: Connect to slope and rate of change
- IDENTITIES: Explain as "different ways of expressing the same relationship"
""",
    
    "functions": """
When teaching Function Graphs concepts:

1. CONNECT TABLES, EQUATIONS, AND GRAPHS: Show multiple representations of the same function.

2. EMPHASIZE INPUT-OUTPUT: Reinforce that each x-value has exactly one y-value in a function.

3. FOCUS ON FEATURES: Draw attention to key features like intercepts, extrema, asymptotes, and symmetry.

4. INTERPRET GRAPHICALLY: Explain what the graph reveals about the situation being modeled.

5. ANALYZE TRANSFORMATIONS: Show how changes to equations affect graph shapes and positions.

6. DEVELOP GRAPH LITERACY: Teach students to extract information from graphs and sketch graphs from descriptions.

7. CONNECT TO RATES OF CHANGE: Relate the shape of graphs to how quickly values are changing.

8. USE TECHNOLOGY APPROPRIATELY: Balance technology use with by-hand sketching to develop conceptual understanding.

For specific function topics:

- LINEAR FUNCTIONS: Connect slope to rate of change and y-intercept to initial value
- QUADRATIC FUNCTIONS: Relate vertex form, factored form, and standard form to different graph features
- EXPONENTIAL FUNCTIONS: Emphasize the constant ratio property and connect to growth/decay scenarios
"""
}

# Interaction pattern prompt
INTERACTION_PATTERN_PROMPT = """
Structure your teaching interactions following this classroom-like pattern:

1. OPENING: Begin with a brief introduction to the topic and its relevance.
   Example: "Today we're going to learn about quadratic equations, which are incredibly useful for modeling many real-world situations like projectile motion."

2. ACTIVATION: Activate prior knowledge by connecting to previously learned concepts.
   Example: "Remember how we worked with linear equations? Quadratic equations are similar, but they include x² terms, which creates that curved shape we call a parabola."

3. INSTRUCTION: Present new information in small, digestible chunks.
   Example: "A quadratic equation has the form ax² + bx + c = 0, where a, b, and c are constants and a ≠ 0. Let's break down what each part means..."

4. MODELING: Demonstrate problem-solving with explicit thinking.
   Example: "Let me show you how to solve x² - 5x + 6 = 0. First, I'll try factoring because the coefficients look friendly..."

5. GUIDED PRACTICE: Walk through examples with increasing student participation.
   Example: "Let's solve this next one together. What would be your first step for x² + 7x + 12 = 0?"

6. CHECKING: Verify understanding before moving on.
   Example: "Before we continue, can you tell me what the values of a, b, and c are in the equation x² - 3x - 4 = 0?"

7. INDEPENDENT PRACTICE: Provide problems for students to solve.
   Example: "Now try solving these equations on your own: [problems]. I'll guide you if you get stuck."

8. SUMMARY: Recap key points and connect to the bigger picture.
   Example: "Today we've learned how to solve quadratic equations by factoring. This is just one method, and next time we'll explore the quadratic formula."

9. PREVIEW: Briefly mention what comes next in the learning sequence.
   Example: "In our next lesson, we'll see how to use these equations to model real-world problems like the height of a thrown ball."
"""

def get_complete_system_prompt(topic=None):
    """
    Generate a complete system prompt based on the detected topic.
    
    Args:
        topic (str, optional): The math topic being discussed. Defaults to None.
        
    Returns:
        str: The complete system prompt.
    """
    prompt_parts = [BASE_TEACHER_PROMPT, PEDAGOGICAL_APPROACH_PROMPT, INTERACTION_PATTERN_PROMPT]
    
    # Add topic-specific prompt if available
    if topic and topic.lower() in TOPIC_PROMPTS:
        prompt_parts.insert(2, TOPIC_PROMPTS[topic.lower()])
    
    return "\n\n".join(prompt_parts)

def detect_topic(query):
    """
    Detect the math topic from the student's query.
    
    Args:
        query (str): The student's question or request.
        
    Returns:
        str: The detected topic or None if no topic is detected.
    """
    query_lower = query.lower()
    
    # Simple keyword matching for topic detection
    if any(term in query_lower for term in ["equation", "variable", "expression", "factor", "quadratic"]):
        return "algebra"
    elif any(term in query_lower for term in ["triangle", "circle", "angle", "shape", "perimeter", "area"]):
        return "geometry"
    elif any(term in query_lower for term in ["sin", "cos", "tan", "angle", "radian", "degree"]):
        return "trigonometry"
    elif any(term in query_lower for term in ["function", "graph", "plot", "curve", "input", "output"]):
        return "functions"
    
    return None
```

### 4.3 Modified Main Application

```python
# app.py
import os
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama

# Import custom modules
from chunker import MathTeacherChunker
from prompts import get_complete_system_prompt, detect_topic
from utils import load_math_content

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit
st.title("Math Teacher RAG System")
st.write("Learn math concepts with a virtual teacher that explains like a real classroom teacher")

# Input for model selection
model_option = st.sidebar.selectbox(
    "Select LLM Model",
    ["qwen-qwq-32b", "Other models..."]
)

# Initialize LLM
if model_option == "qwen-qwq-32b":
    llm = Ollama(model="qwen-qwq-32b")
else:
    # Fallback to other models
    api_key = st.text_input("Enter your Groq API key:", type="password")
    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name="deepseek-r1-distill-llama-70b")

# Session management
session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

# Load and process math content
if 'vectorstore' not in st.session_state:
    with st.spinner("Loading math content..."):
        # Load math content from data directory
        documents = load_math_content("./data")
        
        # Use custom chunker
        chunker = MathTeacherChunker()
        chunks = chunker.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        st.session_state.vectorstore = vectorstore

# Set up retriever
retriever = st.session_state.vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": 5, "fetch_k": 10}  # Retrieve more, then filter for diversity
)

# Contextualize question system prompt
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Get session history
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# User interface
user_input = st.text_input("Your question:")

if user_input:
    # Detect topic from user input
    topic = detect_topic(user_input)
    
    # Get appropriate system prompt
    system_prompt = get_complete_system_prompt(topic)
    
    # Create QA prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create QA chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # Create conversational RAG chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    # Process user input
    with st.spinner("Thinking like a math teacher..."):
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            },
        )
    
    # Display response
    st.write("Teacher:", response['answer'])
    
    # Display chat history
    with st.expander("Chat History"):
        st.write(session_history.messages)
```

### 4.4 Utility Functions

```python
# utils.py
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
    
    # Copy sample content files to appropriate directories
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
```

## 5. Implementation Workflow

1. Create the project directory structure
2. Implement the utility functions
3. Implement the custom chunker
4. Implement the prompt management system
5. Implement the main application
6. Test and refine the system

By implementing these improvements, the RAG system will better simulate how a teacher presents math concepts in a classroom, providing a more coherent and effective learning experience for high school students.
