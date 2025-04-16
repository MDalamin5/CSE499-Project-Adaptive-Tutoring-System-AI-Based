# Running Instructions for Math Teacher RAG System

This guide provides step-by-step instructions for running and testing the Math Teacher RAG system.

## Prerequisites

Before running the system, ensure you have the following prerequisites installed:

1. Python 3.8 or higher
2. Ollama (for running local LLMs)
3. Required Python packages

## Installation Steps

### 1. Install Required Python Packages

Run the following command to install all required packages:

```bash
pip install langchain langchain-community langchain-core langchain-huggingface langchain-chroma streamlit chromadb pydantic sentence-transformers
```

### 2. Install Ollama

Ollama is required to run the LLM locally. Follow the installation instructions for your operating system:

- **Linux**: 
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

- **macOS**: 
  Download from https://ollama.com/download

- **Windows**: 
  Download from https://ollama.com/download

### 3. Pull the Required Model

After installing Ollama, pull the model you want to use:

```bash
ollama pull qwen-qwq-32b
```

Alternatively, you can use other models like:
```bash
ollama pull llama2
ollama pull gemma:7b
ollama pull mistral
```

## Running the System

### 1. Navigate to the Project Directory

```bash
cd /home/ubuntu/math_teacher_rag
```

### 2. Start the Streamlit Application

```bash
streamlit run app.py
```

This will start the web interface and open it in your default browser. If it doesn't open automatically, you can access it at http://localhost:8501.

### 3. Using the Web Interface

1. **Initialize the Model**:
   - Select the model from the dropdown in the sidebar (e.g., "qwen-qwq-32b")
   - Click the "Initialize Model" button
   - Wait for the model to initialize (this may take a minute)

2. **Load the Content**:
   - Click the "Load Content" button in the sidebar
   - Wait for the content to be processed and indexed

3. **Start a Conversation**:
   - Enter a session ID (or use the default)
   - Type a math question in the input field and press Enter
   - The system will respond with a teacher-like explanation

4. **Continue the Conversation**:
   - Ask follow-up questions to explore the topic further
   - View the conversation history by expanding the "Chat History" section

## Testing the System

To test the system's capabilities, try the following sample questions:

### Algebra Questions:
- "What are quadratic equations and how do I solve them?"
- "Can you explain how to factor polynomials?"
- "How do I solve systems of linear equations?"

### Geometry Questions:
- "What is the Pythagorean theorem and how do I use it?"
- "How do I calculate the area of different shapes?"
- "Can you explain the properties of triangles?"

### Trigonometry Questions:
- "What are sine, cosine, and tangent?"
- "How do I use trigonometric functions to solve problems?"
- "Can you explain the unit circle?"

### Function Questions:
- "What is a function and how do I graph it?"
- "How do transformations affect function graphs?"
- "Can you explain domain and range?"

## Troubleshooting

### Common Issues and Solutions

1. **Model Initialization Fails**:
   - Ensure Ollama is running in the background
   - Check if the model has been pulled correctly
   - Try restarting Ollama service

2. **Content Loading Fails**:
   - Verify that the content files exist in the data directory
   - Check file permissions
   - Ensure the file format is correct (markdown)

3. **Slow Responses**:
   - This is normal for the first few queries as the model warms up
   - Consider using a smaller model if responses are consistently slow
   - Reduce the number of retrieved chunks in app.py

4. **Out of Memory Errors**:
   - Close other memory-intensive applications
   - Use a smaller model
   - Reduce the chunk size in the chunker.py file

5. **Irrelevant Responses**:
   - Try rephrasing your question
   - Check if the topic is correctly detected (shown in sidebar)
   - Ensure the content files contain relevant information

## Advanced Configuration

### Modifying the Chunking Strategy

To adjust how content is chunked, edit the `chunker.py` file:
- Change pattern matching for different section types
- Adjust metadata extraction
- Modify chunk size considerations

### Customizing Teacher Prompts

To modify the teaching style, edit the `prompts.py` file:
- Adjust the base teacher identity prompt
- Modify pedagogical approach instructions
- Update topic-specific teaching strategies
- Change the interaction pattern

### Enhancing Topic Detection

To improve topic detection, edit the `detect_topic` function in `prompts.py`:
- Add more keywords for each topic
- Implement more sophisticated detection logic
- Add support for additional topics

## Conclusion

The Math Teacher RAG system is now set up and ready to use. It provides a classroom-like teaching experience for high school math topics, simulating how a real teacher would explain concepts, provide examples, and guide students through problem-solving.

If you encounter any issues not covered in this guide, please check the documentation files in the `/home/ubuntu/documentation/` directory for more detailed information about the system architecture and components.
