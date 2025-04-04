import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Initialize LLM and Prompt Template ---
@st.cache_resource
def setup_llm():
    model = ChatGroq(model_name="qwen-2.5-32b", temperature=0.1, groq_api_key=groq_api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
You are an AI math tutor specializing in high school algebra. Your goal is to help students understand and solve math problems step-by-step.

**Your Process:**

1.  **Receive the Problem:** The student will give you an equation or math problem.
2.  **Break it Down:** Solve the problem one step at a time. Explain *why* you are doing each step clearly and concisely. Focus on the underlying mathematical principles.
3.  **Ask for Understanding:** After each step, *always* ask the student if they understand the step. Provide options like:
    *   "Do you understand this step?"
    *   "Does this make sense?"
    *   "Are you following along?"
4.  **Respond to Feedback:**
    *   **If the student understands:** Move on to the next step.
    *   **If the student doesn't understand or asks for a hint:** Provide a clear, concise explanation or a relevant hint. Do *not* just repeat the same information. Try a different explanation or a simpler example.
    *   **If the student asks for an alternative method:** Provide a different approach to solving the problem, explaining the reasoning behind it.
5.  **Final Solution:** Once you've reached the final solution, clearly state the answer.
6.  **Offer Next Steps:** Ask if the student wants to try another problem or has any further questions.
7.  **Do not solve whole problem in one response just give response as one step at a time. always ask the student understand or not if they understand then next step otherwise give hint.**

Now let's begin!
"""
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("system",  # Add a placeholder for the adaptive message
             "{{adaptive_message}}")
        ]
    )
    return model, prompt

model, prompt_template = setup_llm()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello! I'm ready to help you with algebra.")]

if "correct" not in st.session_state:
    st.session_state.correct = False

if "attempts_this_step" not in st.session_state:
    st.session_state.attempts_this_step = 0

if "hint_used" not in st.session_state:
    st.session_state.hint_used = False

if "response_time" not in st.session_state:
    st.session_state.response_time = None

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "adaptive_message" not in st.session_state:
    st.session_state.adaptive_message = ""

if "problem" not in st.session_state:
     st.session_state.problem = { #example problem
        "question": "Solve for x: 2x + 3 = 7",
        "correct_answer": "2",
        "difficulty": "Easy",
        "topic": "Solving Linear Equations"
    }

# --- Helper Functions ---
def get_response(prompt, history, adaptive_message=""):
    """Gets the LLM's response using Langchain."""
    chain = prompt | model
    response = chain.invoke({"messages": history, "adaptive_message": adaptive_message})
    return response.content

def check_answer(student_answer, correct_answer):
    """Checks if the student's answer is correct."""
    try:
        student_answer = float(student_answer)
        correct_answer = float(correct_answer)
        return abs(student_answer - correct_answer) < 1e-6  # Account for floating-point errors
    except ValueError:
        return student_answer.strip().lower() == correct_answer.strip().lower()

def generate_adaptive_message(student_performance, contextual_parameters):
    """Generates an adaptive message based on student performance and context."""

    if (not student_performance["understand"] and
        student_performance["attempts_this_step"] >= 2):
        if not student_performance["hint_used"]:
            return "Stuck?  Remember to isolate the variable on one side of the equation."  # provide hint
        else:
            return "Let's try a simpler equation: x + 5 = 10. Can you solve for x?"  # Simplify

    elif (student_performance["understand"] and
          student_performance["attempts_this_step"] >= 4):
        return "Just to be sure, let's quickly review the concept of isolating variables."  # Reinforce Concept

    elif (not student_performance["correctness"] and
          contextual_parameters["error_type"] == "Conceptual Error"):
        return "It seems like there's a misunderstanding of the underlying concept. Let's review the definition of a linear equation."  # explain concenpt

    elif (not student_performance["understand"] and
          contextual_parameters["previous_topic_performance"].get("Fractions") == "Poor" and
          contextual_parameters["topic"] == "Solving Linear Equations"):
        return "Before we continue, let's refresh your skill on Fractions, its important for this topic."  # review preequsite topics

    elif student_performance["understand"]:
        return "Great job! You are in right way."  # encourage
    else:
        return ""  # defualt message


# --- Streamlit UI ---
st.title("AI Algebra Tutor")

# Sidebar for Tracking Output
with st.sidebar:
    st.header("Tracking Information")
    st.write(f"Correct: {st.session_state.correct}")
    st.write(f"Attempts This Step: {st.session_state.attempts_this_step}")
    st.write(f"Hint Used: {st.session_state.hint_used}")
    st.write(f"Response Time: {st.session_state.response_time}")
    st.write(f"Start Time: {st.session_state.start_time}")
    st.write(f"Adaptive Message: {st.session_state.adaptive_message}")
    st.subheader("Conversation History (Session State)")
    st.write(st.session_state.messages)

# Display chat messages from history
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)

# Get user input
if prompt := st.chat_input(st.session_state.problem["question"]): #show the problem
    # Display user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.start_time = st.session_state.get("start_time") or time.time()

    #check the answer
    st.session_state.correct = check_answer(prompt, st.session_state.problem["correct_answer"]) #store correct value to "st.session_state.correct"

    #generate student performance and contextual parameter
    student_performance = {
        "correctness": st.session_state.correct,
        "attempts_this_step": st.session_state.attempts_this_step,
        "understand": False, #get from next step
        "response_time":  st.session_state.response_time,
        "hint_used": st.session_state.hint_used,
    }

    contextual_parameters = {
        "problem_difficulty": st.session_state.problem["difficulty"],
        "topic": st.session_state.problem["topic"],
        "previous_topic_performance": {"Fractions": "Good", "Order of Operations": "Fair"},
        "error_type": "Conceptual Error",
    }

    # Generate Adaptive Message
    st.session_state.adaptive_message = generate_adaptive_message(student_performance, contextual_parameters)
    adaptive_message = st.session_state.adaptive_message

    # Get assistant response with adaptive message
    full_prompt = prompt_template.format_messages(messages=st.session_state.messages)
    response = get_response(prompt_template, st.session_state.messages, adaptive_message)

    # Display assistant message
    st.session_state.messages.append(AIMessage(content=response))
    with st.chat_message("assistant"):
        st.markdown(response)

    #update st.session_state.response_time
    if st.session_state.start_time:
        st.session_state.response_time = time.time() - st.session_state.start_time

    # Update attempts number after user message
    st.session_state.attempts_this_step += 1

    #radio button to track understand or not
    understand = st.radio(
    "Do you understand?",
    ("Yes", "No"))
    if understand == "Yes":
        st.session_state.correct = True #its mean understand
        st.session_state.understand = True
    elif understand == "No":
        st.session_state.correct = False # its mean not understand
        st.session_state.understand = False

    # Refresh Sidebar
    with st.sidebar:
        st.header("Tracking Information")
        st.write(f"Correct: {st.session_state.correct}")
        st.write(f"Attempts This Step: {st.session_state.attempts_this_step}")
        st.write(f"Hint Used: {st.session_state.hint_used}")
        st.write(f"Response Time: {st.session_state.response_time}")
        st.write(f"Start Time: {st.session_state.start_time}")
        st.write(f"Adaptive Message: {st.session_state.adaptive_message}")
        st.subheader("Conversation History (Session State)")
        st.write(st.session_state.messages)