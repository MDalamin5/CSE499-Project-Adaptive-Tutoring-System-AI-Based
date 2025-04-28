# Usage Instructions

## System Requirements

- Python 3.8+
- Required packages: langchain, streamlit, chromadb, huggingface_hub, ollama

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/math_teacher_rag.git
cd math_teacher_rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

The requirements.txt file should contain:

```
langchain>=0.1.0
langchain-community>=0.0.10
langchain-core>=0.1.10
langchain-groq>=0.1.0
langchain-huggingface>=0.0.2
langchain-chroma>=0.0.10
streamlit>=1.30.0
chromadb>=0.4.22
pydantic>=2.5.0
```

### 3. Set Up Educational Content

The system comes with pre-loaded educational content for high school math topics. The content is organized in the following directory structure:

```
/data/
├── algebra/
│   └── algebra.md
├── geometry/
│   └── geometry.md
├── trigonometry/
│   └── trigonometry.md
└── functions/
    └── functions.md
```

You can add your own content by:

1. Creating markdown files following the structured format
2. Placing files in the appropriate topic directory
3. Restarting the application to process new content

## Running the Application

### 1. Start the Application

```bash
streamlit run app.py
```

### 2. Access the Web Interface

Open a browser and navigate to:
- http://localhost:8501

## Using the System

### 1. Initial Setup

When you first access the application, you'll see:

1. A title "Math Teacher RAG System"
2. A model selection dropdown in the sidebar
3. A session ID input field
4. A question input field

### 2. Select a Model

From the sidebar, select the LLM model to use:
- "qwen-qwq-32b" (recommended for best performance)
- Other models (may require API keys)

### 3. Enter a Session ID

Enter a session ID to maintain conversation history across interactions. You can:
- Use "default_session" for general use
- Create topic-specific sessions like "algebra_session" or "geometry_session"
- Create student-specific sessions like "student1_session"

### 4. Ask a Question

Type a math question or topic you want to learn in the input field and press Enter. For example:
- "I want to learn about quadratic equations"
- "Can you explain the Pythagorean theorem?"
- "How do I solve this equation: 2x + 3 = 7?"
- "What are sine and cosine used for?"

### 5. Interact with the Teacher

After asking a question:
1. The system will detect the math topic
2. Retrieve relevant educational content
3. Generate a teacher-like response
4. Display the response in the chat interface

Continue the conversation by asking follow-up questions. The system maintains conversation history to provide contextually relevant responses.

### 6. View Conversation History

Click on the "Chat History" expander to view the full conversation history for the current session.

## Example Interactions

### Example 1: Learning a New Topic

**Student**: "I want to learn about trigonometry"

**Teacher**: "Great! I'd be happy to introduce you to trigonometry. Before we dive in, let me check if you're familiar with some prerequisites. Do you have a good understanding of right triangles and the Pythagorean theorem?"

**Student**: "Yes, I understand those concepts"

**Teacher**: "Excellent! That gives us a solid foundation to build on.

Trigonometry is the study of relationships between the angles and sides of triangles. It's incredibly useful in fields like engineering, physics, architecture, and even computer graphics.

Let's start with the three basic trigonometric ratios in a right triangle:

1. Sine (sin): The ratio of the opposite side to the hypotenuse
   sin(θ) = opposite / hypotenuse

2. Cosine (cos): The ratio of the adjacent side to the hypotenuse
   cos(θ) = adjacent / hypotenuse

3. Tangent (tan): The ratio of the opposite side to the adjacent side
   tan(θ) = opposite / adjacent

A helpful way to remember these is the acronym SOH-CAH-TOA:
- SOH: Sine = Opposite / Hypotenuse
- CAH: Cosine = Adjacent / Hypotenuse
- TOA: Tangent = Opposite / Adjacent

Let's look at a simple example. In a right triangle, if the angle θ is 30° and the hypotenuse is 10 cm, we can find the opposite side using sine:

sin(30°) = opposite / 10
opposite = 10 × sin(30°)
opposite = 10 × 0.5
opposite = 5 cm

Would you like to try a practice problem using these concepts?"

### Example 2: Solving a Specific Problem

**Student**: "How do I solve x² - 5x + 6 = 0?"

**Teacher**: "Let's solve this quadratic equation: x² - 5x + 6 = 0.

This equation is in the standard form ax² + bx + c = 0, where a = 1, b = -5, and c = 6.

For this equation, I'll use the factoring method since the coefficients look friendly.

Step 1: I need to find two numbers that multiply to give 6 (which is c) and add to give -5 (which is b).
The factors of 6 are: 1 × 6 = 6 and 2 × 3 = 6
Checking their sums:
1 + 6 = 7 (not -5)
2 + 3 = 5 (not -5)

Since I need a sum of -5, I need negative numbers:
-1 × -6 = 6, and -1 + (-6) = -7 (not -5)
-2 × -3 = 6, and -2 + (-3) = -5 (this works!)

Step 2: Rewrite the middle term using these numbers:
x² - 2x - 3x + 6 = 0

Step 3: Factor by grouping:
x(x - 2) - 3(x - 2) = 0
(x - 3)(x - 2) = 0

Step 4: Apply the zero product property:
Either (x - 3) = 0 or (x - 2) = 0
So x = 3 or x = 2

Step 5: Let's verify our answers:
For x = 3: 3² - 5(3) + 6 = 9 - 15 + 6 = 0 ✓
For x = 2: 2² - 5(2) + 6 = 4 - 10 + 6 = 0 ✓

Therefore, the solutions to x² - 5x + 6 = 0 are x = 2 and x = 3.

Would you like to try another quadratic equation, or shall we explore a different method for solving quadratics, like the quadratic formula?"

## Extending the System

### Adding New Content

1. Create markdown files following the structured format with sections for:
   - Prerequisites
   - Topic overview
   - Core concepts
   - Teaching flow
   - Practice problems
   - Common mistakes
   - Tips and strategies

2. Place files in the appropriate topic directory:
   - `/data/algebra/` for algebra content
   - `/data/geometry/` for geometry content
   - `/data/trigonometry/` for trigonometry content
   - `/data/functions/` for function and graph content

3. Restart the application to process new content

### Customizing Prompts

You can customize the teacher simulation prompts by editing the `prompts.py` file:

1. Modify the base teacher identity prompt to change the teaching persona
2. Adjust the pedagogical approach prompt to emphasize different teaching strategies
3. Edit topic-specific prompts to focus on particular aspects of each math domain
4. Customize the interaction pattern to change the teaching flow

### Performance Optimization

If you experience performance issues:

1. Adjust chunk sizes in the chunker configuration
2. Tune retrieval parameters for better results
3. Optimize embedding model selection
4. Reduce the number of chunks retrieved per query
5. Use a more powerful LLM if available

## Troubleshooting

### Common Issues

1. **Slow Response Times**
   - Reduce the number of retrieved chunks
   - Use a smaller embedding model
   - Check system resources

2. **Irrelevant Responses**
   - Check that content files are properly formatted
   - Adjust the retrieval parameters
   - Ensure topic detection is working correctly

3. **Missing Content**
   - Verify that content files are in the correct directories
   - Check file formatting and structure
   - Restart the application to reprocess content

4. **Model Loading Errors**
   - Ensure the selected model is available
   - Check API keys if using external models
   - Verify system requirements are met

### Getting Help

If you encounter issues not covered in this documentation:

1. Check the GitHub repository for updates and known issues
2. Review the code comments for implementation details
3. Contact the system maintainer for assistance

## Conclusion

The Math Teacher RAG System provides a sophisticated approach to simulating classroom teaching for high school math topics. By following these usage instructions, you can effectively deploy, use, and extend the system to create engaging and effective learning experiences for students.
