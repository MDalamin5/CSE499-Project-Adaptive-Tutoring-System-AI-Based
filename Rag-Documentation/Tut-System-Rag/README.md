# Math Teacher RAG System Documentation

## Overview

The Math Teacher RAG System is an advanced educational tool designed to simulate how teachers teach mathematics in a high school classroom setting. The system uses Retrieval-Augmented Generation (RAG) to provide students with interactive, pedagogically sound math instruction across various topics including Algebra, Geometry, Trigonometry, and Function Graphs.

This documentation provides comprehensive information about the system architecture, implementation details, and usage instructions.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Chunking Strategy](#chunking-strategy)
3. [Prompt Engineering Approach](#prompt-engineering-approach)
4. [Implementation Details](#implementation-details)
5. [Usage Instructions](#usage-instructions)
6. [Maintenance and Extension](#maintenance-and-extension)

## System Architecture

The Math Teacher RAG System consists of the following key components:

### Directory Structure

```
/math_teacher_rag/
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

### Component Interaction

1. **Content Processing Pipeline**:
   - Educational content is loaded from markdown files
   - Custom chunker processes content based on pedagogical structure
   - Chunks are embedded and stored in a vector database

2. **Retrieval System**:
   - User queries are processed to identify the math topic
   - History-aware retriever fetches relevant chunks
   - Metadata-enhanced retrieval prioritizes pedagogically appropriate content

3. **Generation System**:
   - Teacher simulation prompts guide the LLM's response
   - Topic-specific teaching strategies are applied
   - Teaching flow is managed to simulate classroom instruction

4. **User Interface**:
   - Streamlit-based web interface
   - Session management for conversation history
   - Interactive question-answering

### Data Flow

```
User Query → Topic Detection → Retrieval → Context Assembly → 
Teacher Prompt Selection → LLM Generation → Response Formatting → User Display
```

## Chunking Strategy

The system employs a semantic chunking strategy that preserves the pedagogical structure of math instruction, ensuring that related concepts stay together and the teaching flow remains coherent.

### Hierarchical Chunking Structure

1. **Topic-Level Chunks**:
   - Topic overview
   - Prerequisites
   - Learning objectives
   - Topic importance

2. **Concept-Level Chunks**:
   - Concept definitions and explanations
   - Key formulas and properties
   - Teaching methodology
   - Basic examples

3. **Application-Level Chunks**:
   - Practice problems and solutions
   - Common mistakes and misconceptions
   - Real-world applications
   - Advanced examples

4. **Meta-Information Chunks**:
   - Teaching strategies
   - Transition guidance
   - Assessment strategies
   - Differentiation approaches

### Metadata Enhancement

Each chunk includes metadata to improve retrieval relevance:

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

### Implementation Details

The chunking strategy is implemented through the `MathTeacherChunker` class, which:

1. Processes markdown files based on their structure
2. Identifies section boundaries using header patterns
3. Extracts metadata from content
4. Creates semantically meaningful chunks
5. Preserves teaching flow and pedagogical structure

## Prompt Engineering Approach

The system uses a multi-layered prompt architecture to guide the LLM in simulating a math teacher's classroom instruction.

### Prompt Architecture

1. **Base Teacher Identity Layer**:
   - Establishes the AI's role as a math teacher
   - Defines teaching philosophy and approach
   - Sets expectations for student interaction

2. **Pedagogical Approach Layer**:
   - Defines teaching methodologies
   - Outlines instructional strategies
   - Guides content presentation

3. **Content Adaptation Layer**:
   - Guides transformation of formal content into teaching language
   - Structures information sequencing
   - Balances precision with accessibility

4. **Interaction Pattern Layer**:
   - Structures the teaching conversation flow
   - Simulates classroom dialogue patterns
   - Guides pacing and transitions

5. **Assessment & Feedback Layer**:
   - Guides checking for understanding
   - Structures feedback provision
   - Simulates classroom assessment techniques

### Topic-Specific Prompts

The system includes specialized prompts for different math topics:

- **Algebra**: Emphasizes patterns, variables, and equation-solving strategies
- **Geometry**: Focuses on spatial reasoning, visualization, and properties
- **Trigonometry**: Highlights angle relationships, ratios, and applications
- **Function Graphs**: Emphasizes representation, features, and interpretation

### Dynamic Prompt Selection

The system dynamically selects prompts based on:

1. Detected topic from user query
2. Current teaching stage
3. Conversation history
4. Retrieved content type

## Implementation Details

### Custom Chunker Implementation

The `MathTeacherChunker` class processes educational content using regular expressions to identify section boundaries and extract metadata.

Key features:
- Pattern matching for section identification
- Metadata extraction from content
- Hierarchical chunk creation
- Fallback to basic chunking when needed

### Prompt Management System

The prompt system is implemented in `prompts.py` and includes:

- Base teacher identity prompt
- Pedagogical approach prompt
- Topic-specific prompts
- Interaction pattern prompt
- Topic detection function
- Complete prompt assembly function

### Retrieval Enhancement

The retrieval system is enhanced with:

- Maximum Marginal Relevance (MMR) search for diversity
- Metadata filtering for relevance
- History-aware retrieval for context
- Prerequisite awareness

### Teaching Flow Management

The teaching flow is managed through:

- Tracking conversation state
- Structuring responses according to teaching patterns
- Adapting to student questions
- Ensuring progressive skill development

## Usage Instructions

### System Requirements

- Python 3.8+
- Required packages: langchain, streamlit, chromadb, huggingface_hub, ollama

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/math_teacher_rag.git
   cd math_teacher_rag
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare educational content:
   - Place markdown files in the appropriate topic directories
   - Ensure content follows the structured format

### Running the Application

1. Start the application:
   ```
   streamlit run app.py
   ```

2. Access the web interface:
   - Open a browser and navigate to http://localhost:8501

### Using the System

1. Enter a session ID (optional)
2. Type a math question or topic you want to learn
3. Interact with the virtual math teacher
4. Continue the conversation to explore the topic further

### Example Interactions

**Student**: "I want to learn about quadratic equations."

**Teacher**: "Great! Today we're going to explore quadratic equations, which are incredibly useful in mathematics and real-world applications. Before we dive in, let's make sure you're comfortable with some prerequisites. Do you remember working with linear equations like y = mx + b?"

**Student**: "Yes, I understand linear equations."

**Teacher**: "Excellent! So quadratic equations are a step up from linear equations. While linear equations give us straight lines when graphed, quadratic equations create curves called parabolas.

A quadratic equation has the form ax² + bx + c = 0, where a, b, and c are constants and a ≠ 0. The 'a' value determines whether the parabola opens upward or downward.

Let me show you how to solve a simple quadratic equation by factoring. For example, let's solve x² - 5x + 6 = 0.

First, I'll try to find two numbers that multiply to give 6 and add to give -5. Those numbers are -2 and -3.
So I can rewrite the equation as: x² - 2x - 3x + 6 = 0
Factoring: x(x - 2) - 3(x - 2) = 0
(x - 3)(x - 2) = 0

This means either x - 3 = 0 or x - 2 = 0
So x = 3 or x = 2

Let's verify: When x = 3: 3² - 5(3) + 6 = 9 - 15 + 6 = 0 ✓
When x = 2: 2² - 5(2) + 6 = 4 - 10 + 6 = 0 ✓

Would you like to try solving one yourself?"

## Maintenance and Extension

### Adding New Content

1. Create markdown files following the structured format
2. Place files in the appropriate topic directory
3. Restart the application to process new content

### Extending to New Topics

1. Add a new topic directory under `/data`
2. Create content files for the new topic
3. Add a topic-specific prompt in `prompts.py`
4. Update the topic detection function

### Customizing Prompts

1. Edit the prompt templates in `prompts.py`
2. Adjust the teaching strategies for specific topics
3. Modify the interaction patterns as needed

### Performance Optimization

1. Adjust chunk sizes in the chunker configuration
2. Tune retrieval parameters for better results
3. Optimize embedding model selection

## Conclusion

The Math Teacher RAG System provides a sophisticated approach to simulating classroom teaching for high school math topics. By combining semantic chunking, pedagogically-informed prompts, and enhanced retrieval mechanisms, the system delivers a more natural and effective learning experience for students.
