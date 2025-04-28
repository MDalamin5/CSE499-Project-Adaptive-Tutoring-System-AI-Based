import os
import streamlit as st
from typing import List, Dict, Any
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import time

# Set page configuration
st.set_page_config(page_title="Math Teacher RAG System", page_icon="🧮", layout="wide")

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Math Teacher EduMath-RAG"

# os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "") # Get the value if it exists
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Interactive Way Math Problem Solving"


# Constants
DATA_PATH = "./Data"
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class MathTeacherChunker:
    """Custom chunker for math education content"""
    
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


def detect_topic(query):
    """Detect the math topic from the student's query."""
    query_lower = query.lower()
    
    # Simple keyword matching for topic detection
    if any(term in query_lower for term in ["equation", "variable", "expression", "factor", "quadratic", "algebra"]):
        return "algebra"
    elif any(term in query_lower for term in ["triangle", "circle", "angle", "shape", "perimeter", "area", "geometry"]):
        return "geometry"
    elif any(term in query_lower for term in ["sin", "cos", "tan", "angle", "radian", "degree", "trigonometry"]):
        return "trigonometry"
    elif any(term in query_lower for term in ["function", "graph", "plot", "curve", "input", "output"]):
        return "functions"
    
    return None


def get_teacher_prompt(topic=None):
    """Generate a complete teacher prompt based on detected topic"""

    # Base teacher identity prompt
    base_prompt = """
    You are an experienced high school math teacher with years of classroom experience teaching students of diverse abilities and learning styles. Your teaching approach is characterized by:

    1. Clear explanations that break down complex concepts into manageable parts elaborated way.
    2. Connecting abstract mathematical ideas to real-world applications elaborated way.
    3. Anticipating common misconceptions and addressing them proactively elaborated way.
    4. Using a variety of examples that progress from simple to complex
    5. Encouraging active participation through questions and problem-solving

    Your goal is to help students not just memorize formulas, but develop a deep conceptual understanding of mathematics. You teach in a supportive, encouraging manner while maintaining high expectations for student learning.

    Use the retrieved mathematical content to guide your teaching, but adapt it to simulate a classroom teaching experience. Present information as if you're speaking directly to a student, using a conversational yet educational tone.
    """

    # Pedagogical approach prompt
    pedagogical_prompt = """
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

    # Interaction pattern prompt
    interaction_prompt = """
    Structure your teaching interactions following this classroom-like pattern:

    1. OPENING: Begin with a brief introduction to the topic and its relevance.
    2. ACTIVATION: Activate prior knowledge by connecting to previously learned concepts.
    3. INSTRUCTION: Present new information in small, digestible chunks.
    4. MODELING: Demonstrate problem-solving with explicit thinking.
    5. GUIDED PRACTICE: Walk through examples with increasing student participation.
    6. CHECKING: Verify understanding before moving on.
    7. INDEPENDENT PRACTICE: Provide problems for students to solve.
    8. SUMMARY: Recap key points and connect to the bigger picture.
    9. PREVIEW: Briefly mention what comes next in the learning sequence.
    """

    # Topic-specific prompts
    topic_prompts = {
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
        """
    }

    # Combine prompts
    prompt_parts = [base_prompt, pedagogical_prompt]

    # Add topic-specific prompt if applicable
    if topic and topic in topic_prompts:
        prompt_parts.append(topic_prompts[topic])

    prompt_parts.append(interaction_prompt)

    # Formatted RAG instruction (modified)
    rag_instruction = """
    Using the retrieved content (provided in the `context` variable), address the student's question about mathematics. Remember to:

    1. Before explaining anything, check for prerequisite knowledge. Ask the student if they have any confusion about the necessary prerequisite concepts. If they do, address those first.
    2. Present concepts in a logical sequence, building from basic definitions to more complex applications.
    3. Provide clear and detailed explanations, supplementing with illustrative examples to solidify understanding.
    4. If teaching a procedure or formula, explicitly explain both how it works (the steps) and why it works (the underlying principles).
    5. If the retrieved content doesn't fully address the student's question, use your expertise as a math teacher to supplement the retrieved information and provide a complete and satisfactory answer.

    The student's question/request is: {input}

    Retrieved Content: {context} - Use this to guide your explanation.
    """

    prompt_parts.append(rag_instruction)

    return "\n\n".join(prompt_parts)


def load_documents():
    """Load PDF documents from the data directory"""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data directory not found: {DATA_PATH}")
        return []
    
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    
    st.info("Loading PDF documents... This may take a moment.")
    documents = loader.load()
    st.success(f"Loaded {len(documents)} documents.")
    
    return documents


def get_or_create_vectorstore(documents):
    """Get existing vector store or create a new one if it doesn't exist"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Check if vector store already exists
    if os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH):
        st.info("Loading existing vector store...")
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        
        # Check if the collection has documents
        if vector_store._collection.count() > 0:
            st.success(f"Loaded existing vector store with {vector_store._collection.count()} chunks.")
            return vector_store
    
    # If we get here, we need to create a new vector store
    if not documents:
        st.error("No documents provided for creating vector store.")
        return None
    
    st.info("Creating new vector store... This may take a while.")
    
    # Process the documents with the MathTeacherChunker
    chunker = MathTeacherChunker()
    chunks = chunker.split_documents(documents)
    
    st.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    vector_store.persist()
    
    st.success(f"Created and saved new vector store with {len(chunks)} chunks.")
    
    return vector_store


def create_multi_query_retriever(llm, retriever, verbose=False):
    """Create a multi-query retriever to improve retrieval performance"""
    return MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )


def create_rag_chain(llm, retriever, topic=None):
    """Create a RAG chain for answering questions"""
    # Create the teacher prompt template
    teacher_prompt = get_teacher_prompt(topic)
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", teacher_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create the QA chain
    qa_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
        document_variable_name="context"
    )
    
    # Create the history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        ChatPromptTemplate.from_messages([
            ("system", "Given a conversation history and the latest user question, formulate a search query that will help retrieve relevant information to answer the user's question. Focus on extracting mathematical concepts, topics, and specific problem details from the conversation."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "Given the above conversation, generate a search query to retrieve relevant mathematical information to answer the user's question."),
        ]),
    )
    
    # Create the final RAG chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )
    
    return rag_chain


def init_session_state():
    """Initialize the session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default_session"


def display_chat_history():
    """Display the chat history in the Streamlit UI"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def main():
    st.title("Math Teacher RAG System 🧮")
    st.markdown("Ask me any math question, and I'll teach you step by step like a real math teacher!")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API key input
        # groq_api_key = st.text_input("Enter Groq API Key:", type="password")
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Session ID
        session_id = st.text_input("Session ID:", value=st.session_state.session_id)
        if session_id != st.session_state.session_id:
            st.session_state.session_id = session_id
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        # Options
        show_retrieved = st.checkbox("Show Retrieved Documents", value=False)
        verbose_retrieval = st.checkbox("Verbose Multi-Query", value=False)
        
        st.divider()
        
        
        
        st.sidebar.markdown("### Related Projects")
        st.sidebar.markdown(
            """
            <style>
            a {
                color: #4CAF50; /* Green color */
                text-decoration: none; /* Remove underline */
            }
            a:hover {
                text-decoration: underline; /* Underline on hover */
            }
            </style>
            <ul>
                <li><a href="https://interactve.streamlit.app/">Interactive Problem Solving</a></li>
                <li><a href="https://solvemate.streamlit.app/">Step-by-Step-Problem Solving</a></li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
        
        
    
    # Set up the LLM and vector store if not already done
    if not st.session_state.llm and groq_api_key:
        try:
            st.session_state.llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name="meta-llama/llama-4-scout-17b-16e-instruct"
            )
            st.info("LLM initialized successfully.")
        except Exception as e:
            st.error(f"Error initializing LLM: {e}")
    
    if not st.session_state.vector_store:
        documents = load_documents()
        st.session_state.vector_store = get_or_create_vectorstore(documents)
    
    if st.session_state.vector_store and not st.session_state.retriever and st.session_state.llm:
        base_retriever = st.session_state.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        st.session_state.retriever = create_multi_query_retriever(
            st.session_state.llm,
            base_retriever,
            verbose=verbose_retrieval
        )
        st.info("Retriever initialized successfully.")
    
    # Display chat history
    display_chat_history()
    
    # Get user input
    user_input = st.chat_input("Ask your math question...")
    
    if user_input and st.session_state.llm and st.session_state.retriever:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Detect math topic
        topic = detect_topic(user_input)
        
        # Create the RAG chain with the appropriate prompts for the topic
        rag_chain = create_rag_chain(st.session_state.llm, st.session_state.retriever, topic)
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Teaching..."):
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.chat_history[:-1]]
                })
                
                # Display retrieved documents if option is enabled
                if show_retrieved and "context" in response:
                    with st.expander("Retrieved Content"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Document {i+1}**: {doc.metadata.get('source', 'Unknown source')}")
                            st.markdown(f"**Topic**: {doc.metadata.get('topic', 'Unknown topic')}")
                            st.markdown(f"**Type**: {doc.metadata.get('chunk_type', 'Unknown type')}")
                            st.markdown(f"**Content**: {doc.page_content[:200]}...")
                            st.divider()
                
                # Display the response
                def stream_data():
                    for word in response['answer'].split(" "):
                        yield word + " "
                        time.sleep(0.0087)
                # st.write(response["answer"])
                st.write_stream(stream_data)
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
    elif user_input:
        # Handle the case where the user inputs something but the system isn't ready
        with st.chat_message("assistant"):
            if not groq_api_key:
                st.error("Please enter your Groq API key in the sidebar to initialize the system.")
            elif not st.session_state.vector_store:
                st.error("Vector store initialization failed. Please check your data directory.")
            else:
                st.error("System initialization error. Please refresh the page and try again.")
    

if __name__ == "__main__":
    main()