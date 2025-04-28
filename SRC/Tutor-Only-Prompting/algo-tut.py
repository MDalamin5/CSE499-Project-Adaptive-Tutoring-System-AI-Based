import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import asyncio  # Import asyncio
from prompt import load_system_prompt
from prompt_v2 import load_system_prompt2

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
st.sidebar.title("Settings")
temperature = st.sidebar.slider("Temperature", max_value=1.0, min_value=0.1)
grq_model = st.sidebar.selectbox("Model", ["qwen-2.5-32b","qwen-2.5-coder-32b"])

# Initialize Langchain components
@st.cache_resource
def setup_llm():
    model = ChatGroq(model_name=grq_model, temperature=temperature) # Lower temperature for more predictable responses
    # system_prompt = load_system_prompt()
    system_prompt = load_system_prompt2()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                system_prompt
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    chain = prompt | model
    return chain

@st.cache_resource(show_spinner=False)
def setup_runnable(_chain):
    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    runnable = RunnableWithMessageHistory(_chain, get_session_history)
    return runnable

# Streamlit UI
st.title("এআই গণিত শিক্ষক")  # AI Math Tutor
st.write("স্বাগতম এআই গণিত শিক্ষকে! আপনি এখানে বীজগণিত, ক্যালকুলাস এবং অন্যান্য গাণিতিক সমস্যা সমাধান করতে পারেন। এছাড়াও, আপনি নতুন গাণিতিক ধারণা শিখতে পারেন। একটি সমস্যা লিখুন অথবা একটি বিষয় সম্পর্কে জিজ্ঞাসা করুন!")


# Use these variables in your CSS or HTML to style the chat messages

# Use these variables in your CSS or HTML to style the chat messages
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




# Get user input
if prompt := st.chat_input("এখানে আপনার গণিত সমস্যা লিখুন..."):  # Enter your math problem here...
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize LLM chain here (only once)
    chain = setup_llm()
    runnable = setup_runnable(chain)

    # Invoke the chain and get the AI response
    try:
        with st.spinner("সমাধান তৈরি করা হচ্ছে..."):
            ai_message = runnable.invoke(
                [HumanMessage(content=prompt)],
                config={"configurable": {"session_id": 'math_session'}}
            )
            # Add AI message to session state
            st.session_state.messages.append({"role": "assistant", "content": ai_message.content})
            with st.chat_message("assistant"):
                # st.markdown(ai_message.content)
                # # st.write_stream(ai_message.content)
                response=ai_message.content
                
                def stream_data():
                    for word in response.split(" "):
                        yield word + " "
                        time.sleep(0.015)

                st.write_stream(stream_data)

    except Exception as e:
        st.error(f"একটি ত্রুটি ঘটেছে: {e}") # An error occurred
        st.error("আপনার Groq API কী এবং ইন্টারনেট সংযোগ পরীক্ষা করুন।") # Please check your Groq API key and internet connection.
        st.stop()