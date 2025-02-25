import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define available topics (in Bangla)
topics = {
    "linear_equations": "সরল সমীকরণ",
    "quadratic_equations": "দ্বিঘাত সমীকরণ",
    "geometry_basics": "জ্যামিতি মূল ধারণা",
    "trigonometry_basics": "ত্রিকোণমিতি মূল ধারণা"
}

# Expanded system prompt (in Bangla)
system_prompt_template = """
আপনি একজন এআই গণিত শিক্ষক, যিনি উচ্চ বিদ্যালয়ের গণিত শিক্ষাদানে বিশেষজ্ঞ। আপনার প্রধান কাজ হল শিক্ষার্থীদের নতুন গাণিতিক ধারণা শেখানো এবং তাদের সমস্যা সমাধানে সাহায্য করা। অনুগ্রহ করে মনে রাখবেন যে আপনার উত্তর সবসময় বাংলা ভাষায় দিতে হবে।

**আপনার প্রক্রিয়া:**

1.  **বিষয় নির্বাচন:** শিক্ষার্থী একটি গাণিতিক বিষয় নির্বাচন করবে যা তারা শিখতে চায়।
2.  **পাঠ পরিকল্পনা:** প্রতিটি পাঠ নিম্নলিখিত অংশে বিভক্ত থাকবে:
    *   ভূমিকা: বিষয়টির একটি সংক্ষিপ্ত পরিচিতি।
    *   মূল ধারণা: বিষয়টির মূল ধারণাগুলোর ব্যাখ্যা।
    *   উদাহরণ: ধারণাগুলো বোঝানোর জন্য উদাহরণ।
    *   অনুশীলনী সমস্যা: শিক্ষার্থীদের সমাধানের জন্য সমস্যা।
    *   সারসংক্ষেপ: মূল বিষয়গুলোর পুনরালোচনা।
3.  **ধাপে ধাপে শিক্ষা:** প্রতিটি বিষয় ধাপে ধাপে শেখান। প্রথমে মূল ধারণাগুলো বুঝিয়ে, তারপর উদাহরণ দিন এবং সবশেষে অনুশীলনী সমস্যা দিন।
4.  **বোঝার জন্য জিজ্ঞাসা:** প্রতিটি ধাপের পরে, শিক্ষার্থী এই ধাপটি বুঝতে পেরেছে কিনা জিজ্ঞাসা করুন। আপনি এই ধরনের প্রশ্ন করতে পারেন:
    *   "আপনি কি এই ধাপটি বুঝতে পেরেছেন?"
    *   "এটা কি বোধগম্য?"
    *   "আপনি কি অনুসরণ করতে পারছেন?"
5.  **প্রতিক্রিয়া জানানো:**
    *   যদি শিক্ষার্থী বুঝতে পারে: পরবর্তী ধাপে যান।
    *   যদি শিক্ষার্থী বুঝতে না পারে বা কোনো ইঙ্গিত চায়: একটি স্পষ্ট, সংক্ষিপ্ত ব্যাখ্যা বা প্রাসঙ্গিক ইঙ্গিত দিন। অন্য একটি ব্যাখ্যা বা একটি সহজ উদাহরণ চেষ্টা করুন।
    *   যদি শিক্ষার্থী অন্য কোনো পদ্ধতির জন্য জিজ্ঞাসা করে: সমস্যাটি সমাধানের জন্য একটি ভিন্ন পদ্ধতি দিন এবং এর পিছনের যুক্তি ব্যাখ্যা করুন।
6.  **অনুশীলনী সমস্যা:** শিক্ষার্থীদের অনুশীলনী সমস্যা দিন এবং তাদের সমাধান করতে সহায়তা করুন। প্রতিটি ধাপ বুঝিয়ে দিন এবং প্রয়োজনে ইঙ্গিত দিন।
7.  **সবসময় বাংলা ভাষায় উত্তর দিন।**

এখন, শিক্ষার্থী "{topic}" সম্পর্কে জানতে চায়। আপনি কিভাবে শুরু করবেন? প্রথমে বিষয়টির একটি সংক্ষিপ্ত পরিচিতি দিন।
"""

# Initialize Langchain components
@st.cache_resource
def setup_llm():
    model = ChatGroq(model_name="qwen-2.5-32b", temperature=0.1) # Lower temperature for more predictable responses
    return model

@st.cache_resource(show_spinner=False)
def setup_runnable(_model, system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                system_prompt
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    chain = prompt | _model  # Use _model here
    store = {}

    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    runnable = RunnableWithMessageHistory(chain, get_session_history)
    return runnable

# Streamlit UI
st.title("এআই গণিত শিক্ষক")  # AI Math Tutor

# Topic selection
selected_topic = st.selectbox("আপনি কোন বিষয়ে শিখতে চান?",  # Which topic do you want to learn?
                               options=topics.keys(),
                               format_func=lambda x: topics[x])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Dynamic system prompt based on selected topic
system_prompt = system_prompt_template.format(topic=topics[selected_topic])

# Initialize LLM chain and runnable (only once)
model = setup_llm()
runnable = setup_runnable(model, system_prompt)

# Get user input
if prompt := st.chat_input("এখানে আপনার প্রশ্ন লিখুন..."):  # Enter your question here...
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the chain and get the AI response
    try:
        ai_message = runnable.invoke(
            [HumanMessage(content=prompt)],
            config={"configurable": {"session_id": 'math_session'}}
        )
        # Add AI message to session state
        st.session_state.messages.append({"role": "assistant", "content": ai_message.content})
        with st.chat_message("assistant"):
            st.markdown(ai_message.content)

    except Exception as e:
        st.error(f"একটি ত্রুটি ঘটেছে: {e}") # An error occurred
        st.error("আপনার Groq API কী এবং ইন্টারনেট সংযোগ পরীক্ষা করুন।") # Please check your Groq API key and internet connection.
        st.stop()