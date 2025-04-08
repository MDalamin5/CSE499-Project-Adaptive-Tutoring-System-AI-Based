import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ----- Adaptive Prompt Generator Setup -----
class AdaptivePromptResponse(BaseModel):
    prompt: str

parser = PydanticOutputParser(pydantic_object=AdaptivePromptResponse)

adaptive_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You're an educational prompt designer. Generate a motivational and clear SYSTEM prompt for an AI tutor based on student feedback."),
    ("human", "Feedback: {feedback}\nOriginal Student Input: {student_input}\nCreate a helpful system prompt for the tutor.")
])

adaptive_prompt_chain = adaptive_prompt_template | ChatGroq(model_name="qwen-2.5-32b", temperature=0.4) | parser

def generate_adaptive_prompt(feedback: str, student_input: str) -> str:
    response = adaptive_prompt_chain.invoke({
        "feedback": feedback,
        "student_input": student_input
    })
    return response.prompt

# ----- LLM Tutor Setup with Static System Prompt -----
@st.cache_resource
def setup_llm(system_prompt: str):
    model = ChatGroq(model_name="qwen-2.5-32b", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt | model

@st.cache_resource(show_spinner=False)
def setup_runnable(_chain):
    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(_chain, get_session_history)

# ----- Static System Prompt -----
DEFAULT_SYSTEM_PROMPT = """
You are an AI math tutor specializing in high school algebra. Your goal is to help students understand and solve math problems step-by-step.

**Your Process:**

1. Receive the Problem: The student will give you an equation or math problem.
2. Break it Down: Solve the problem one step at a time. Explain *why* you are doing each step clearly and concisely.
3. Ask for Understanding: After each step, *always* ask the student if they understand.
4. Respond to Feedback:
    * If the student understands: Move on.
    * If not: Give a simpler explanation or hint.
    * If they ask for a different method: Give one with reasoning.
5. Final Solution: Clearly state the answer.
6. Offer Next Steps: Ask if the student wants another problem.
7. **Do not solve the whole problem at once. Always pause for student confirmation.**

**Example:**
Student: Solve 2x + 3 = 7  
You: Subtract 3 from both sides → 2x = 4. Do you understand?
"""

# ----- Streamlit UI -----
st.title("AI Math Tutor (Adaptive Prompting System)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

# Display message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Enter your math problem or question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step 1: Get feedback type (simulate with selectbox or build classifier later)
    feedback = st.selectbox("How did the student respond previously?", ["correct", "partially_correct", "incorrect"])

    # Step 2: Generate adaptive prompt
    try:
        adapted_prompt = generate_adaptive_prompt(feedback, prompt)
        st.session_state.system_prompt = adapted_prompt
        st.success("✅ Adaptive prompt successfully generated.")
    except Exception as e:
        st.warning(f"⚠️ Could not generate adaptive prompt. Using default. Error: {e}")
        adapted_prompt = DEFAULT_SYSTEM_PROMPT

    # Step 3: Setup chain and invoke tutor
    chain = setup_llm(st.session_state.system_prompt)
    runnable = setup_runnable(chain)

    try:
        ai_message = runnable.invoke(
            [HumanMessage(content=prompt)],
            config={"configurable": {"session_id": "math_session"}}
        )
        st.session_state.messages.append({"role": "assistant", "content": ai_message.content})
        with st.chat_message("assistant"):
            st.markdown(ai_message.content)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()
