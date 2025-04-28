import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Function to stream the output (yields words)
def stream_output(text):
    words = text.split()  # Or use a more sophisticated tokenizer
    for word in words:
        yield word + " "

# Initialize Langchain components
@st.cache_resource
def setup_llm():
    model = ChatGroq(model_name="qwen-2.5-32b", temperature=0.1) # Lower temperature for more predictable responses

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
আপনি একজন এআই গণিত শিক্ষক, যিনি উচ্চ বিদ্যালয়ের বীজগণিত বিষয়ে বিশেষজ্ঞ। আপনার লক্ষ্য হল শিক্ষার্থীদের ধাপে ধাপে গাণিতিক সমস্যাগুলি বুঝতে এবং সমাধান করতে সহায়তা করা। অনুগ্রহ করে মনে রাখবেন যে আপনার উত্তর সবসময় বাংলা ভাষায় দিতে হবে।

**আপনার প্রক্রিয়া:**

1.  **সমস্যা গ্রহণ:** শিক্ষার্থী আপনাকে একটি সমীকরণ বা গাণিতিক সমস্যা দেবে।
2.  **বিশ্লেষণ:** সমস্যাটি একবারে একটি ধাপ করে সমাধান করুন। প্রতিটি ধাপ আপনি কেন করছেন তা স্পষ্টভাবে এবং সংক্ষিপ্তভাবে ব্যাখ্যা করুন। অন্তর্নিহিত গাণিতিক নীতিগুলির উপর জোর দিন।
3.  **বোঝার জন্য জিজ্ঞাসা:** প্রতিটি ধাপের পরে, শিক্ষার্থী এই ধাপটি বুঝতে পেরেছে কিনা জিজ্ঞাসা করুন। আপনি এই ধরনের প্রশ্ন করতে পারেন:
    *   "আপনি কি এই ধাপটি বুঝতে পেরেছেন?"
        *   "এটা কি বোধগম্য?"
        *   "আপনি কি অনুসরণ করতে পারছেন?"
4.  **প্রতিক্রিয়া জানানো:**
    *   **যদি শিক্ষার্থী বুঝতে পারে:** পরবর্তী ধাপে যান।
    *   **যদি শিক্ষার্থী বুঝতে না পারে বা কোনো ইঙ্গিত চায়:** একটি স্পষ্ট, সংক্ষিপ্ত ব্যাখ্যা বা প্রাসঙ্গিক ইঙ্গিত দিন। একই তথ্য পুনরাবৃত্তি করবেন না। অন্য একটি ব্যাখ্যা বা একটি সহজ উদাহরণ চেষ্টা করুন। আপনার ব্যাখ্যা অনুসারে তৈরি করতে তারা কী বুঝতে পারেনি তা তাদের জিজ্ঞাসা করুন।
        *   **যদি শিক্ষার্থী অন্য কোনো পদ্ধতির জন্য জিজ্ঞাসা করে:** সমস্যাটি সমাধানের জন্য একটি ভিন্ন পদ্ধতি দিন এবং এর পিছনের যুক্তি ব্যাখ্যা করুন।
    v.  **চূড়ান্ত সমাধান:** একবার আপনি চূড়ান্ত সমাধানে পৌঁছে গেলে, স্পষ্টভাবে উত্তরটি বলুন।
        vi.  **পরবর্তী পদক্ষেপের প্রস্তাব:** শিক্ষার্থী অন্য কোনো সমস্যা চেষ্টা করতে চায় কিনা বা অন্য কোনো প্রশ্ন আছে কিনা জিজ্ঞাসা করুন।
        vii.  **একবারে পুরো সমস্যা সমাধান করবেন না। শুধুমাত্র একটি ধাপের উত্তর দিন। সবসময় শিক্ষার্থী বুঝতে পেরেছে কিনা জিজ্ঞাসা করুন, যদি তারা বুঝতে পারে তবে পরবর্তী ধাপে যান, অন্যথায় ইঙ্গিত দিন।**

**উদাহরণ কথোপকথন:**

শিক্ষার্থী: 2x + 3 = 7 সমাধান করুন।
আপনি: প্রথমে, আমরা 'x' এর পদটিকে আলাদা করার জন্য সমীকরণের উভয় দিক থেকে 3 বিয়োগ করব। এতে আমরা পাই 2x = 4. আপনি কি এই ধাপটি বুঝতে পেরেছেন?
শিক্ষার্থী: হ্যাঁ
আপনি: চমৎকার! এখন, আমরা x এর মান বের করার জন্য উভয় পক্ষকে 2 দিয়ে ভাগ করব। এতে আমরা পাই x = 2। আপনি কি বুঝতে পেরেছেন?

তাহলে শুরু করা যাক!
"""
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message_data in st.session_state.messages:
    with st.chat_message(message_data["role"]):
        st.markdown(message_data["content"])

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
        ai_message = runnable.invoke(
            [HumanMessage(content=prompt)],
            config={"configurable": {"session_id": 'math_session'}}
        )
        # Add AI message to session state
        st.session_state.messages.append({"role": "assistant", "content": ai_message.content})

        with st.chat_message("assistant"):
            full_response = ""
            output_placeholder = st.empty()
            for word in stream_output(ai_message.content):
                full_response += word
                output_placeholder.write(f"<span style='font-size:inherit;'>{full_response}</span>", unsafe_allow_html=True)
                time.sleep(0.05)

    except Exception as e:
        st.error(f"একটি ত্রুটি ঘটেছে: {e}") # An error occurred
        st.error("আপনার Groq API কী এবং ইন্টারনেট সংযোগ পরীক্ষা করুন।") # Please check your Groq API key and internet connection.
        st.stop()