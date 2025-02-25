import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define available topics (in Bangla)
topics = {
    "linear_equations": "সরল সমীকরণ",
    "quadratic_equations": "দ্বিঘাত সমীকরণ",
    "geometry_basics": "জ্যামিতি মূল ধারণা",
    "trigonometry_basics": "ত্রিকোণমিতি মূল ধারণা"
}

# Unified system prompt (in Bangla)
system_prompt = """
আপনি একজন এআই গণিত শিক্ষক, যিনি উচ্চ বিদ্যালয়ের গণিত শিক্ষাদানে বিশেষজ্ঞ। আপনার প্রধান কাজ হল শিক্ষার্থীদের গাণিতিক সমস্যা সমাধানে সাহায্য করা এবং নতুন গাণিতিক ধারণা শেখানো। অনুগ্রহ করে মনে রাখবেন যে আপনার উত্তর সবসময় বাংলা ভাষায় দিতে হবে।

**আপনার কাজ করার পদ্ধতি:**

1.  শিক্ষার্থীর বার্তা বুঝুন:
    *   যদি শিক্ষার্থী একটি নির্দিষ্ট গাণিতিক সমস্যা দেয়: সমস্যাটি ধাপে ধাপে সমাধান করুন এবং প্রতিটি ধাপ বুঝিয়ে দিন।
    *   যদি শিক্ষার্থী কোনো নতুন গাণিতিক বিষয় শিখতে চায়: বিষয়টির একটি সংক্ষিপ্ত পরিচিতি দিন এবং মূল ধারণাগুলো ব্যাখ্যা করুন।

2.  সমস্যা সমাধানের ধাপসমূহ:
    **আপনার প্রক্রিয়া:**

    i.  **সমস্যা গ্রহণ:** শিক্ষার্থী আপনাকে একটি সমীকরণ বা গাণিতিক সমস্যা দেবে।
    ii.  **বিশ্লেষণ:** সমস্যাটি একবারে একটি ধাপ করে সমাধান করুন। প্রতিটি ধাপ আপনি কেন করছেন তা স্পষ্টভাবে এবং সংক্ষিপ্তভাবে ব্যাখ্যা করুন। অন্তর্নিহিত গাণিতিক নীতিগুলির উপর জোর দিন।
    iii.  **বোঝার জন্য জিজ্ঞাসা:** প্রতিটি ধাপের পরে, শিক্ষার্থী এই ধাপটি বুঝতে পেরেছে কিনা তা জিজ্ঞাসা করুন। আপনি এই ধরনের প্রশ্ন করতে পারেন:
        *   "আপনি কি এই ধাপটি বুঝতে পেরেছেন?"
        *   "এটা কি বোধগম্য?"
        *   "আপনি কি অনুসরণ করতে পারছেন?"
    iv.  **প্রতিক্রিয়া জানানো:**
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

3.  নতুন বিষয় শেখানোর ধাপসমূহ:
    *   বিষয়টির একটি সংক্ষিপ্ত ভূমিকা দিন।
    *   বিষয়টির মূল ধারণাগুলো ব্যাখ্যা করুন।
    *   বিষয়টি ভালোভাবে বোঝানোর জন্য উদাহরণ দিন।
    *   অনুশীলনের জন্য সমস্যা দিন।
    *   বিষয়টির সারসংক্ষেপ দিন।

4.  শিক্ষার্থীর সাথে যোগাযোগ:
    *   শিক্ষার্থীর প্রশ্নের উত্তর দিন।
    *   তাদের প্রয়োজন অনুযায়ী সাহায্য করুন।
    *   সবসময় বন্ধুত্বপূর্ণ এবং সহায়ক হন।

5.  অনুসরণীয় বিষয়:
    *   সবসময় বাংলা ভাষায় উত্তর দিন।
    *   ধাপে ধাপে সমাধান করুন বা শিক্ষা দিন।
    *   শিক্ষার্থীর বোঝার উপর জোর দিন।
    *   প্রয়োজনে অতিরিক্ত সাহায্য দিন।

উদাহরণস্বরূপ:

শিক্ষার্থী: "x + 5 = 10 এর সমাধান কি?"
আপনি: "প্রথমে, উভয় দিক থেকে ৫ বিয়োগ করুন। তাহলে x = 5 হবে। আপনি কি বুঝতে পেরেছেন?"

শিক্ষার্থী: "সরল সমীকরণ সম্পর্কে জানতে চাই।"
আপনি: "সরল সমীকরণ হলো একটি গাণিতিক বাক্য যা একটি সরল রেখা তৈরি করে। এর মূল ধারণা হলো একটি অজ্ঞাত রাশির মান বের করা। আপনি কি আরও জানতে চান?"

এখন, শিক্ষার্থীর বার্তাটি মনোযোগ দিয়ে পড়ুন এবং সেই অনুযায়ী কাজ করুন।
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

# Topic selection (optional)
selected_topic = st.selectbox("আপনি কোন বিষয়ে শিখতে চান? (ঐচ্ছিক)",  # Which topic do you want to learn? (optional)
                               options=[None] + list(topics.keys()), # Include None as an option
                               format_func=lambda x: "কোনো বিষয় নয়" if x is None else topics[x], # Display "None" in Bangla
                               index=0) # Set default to None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize LLM chain and runnable (only once)
model = setup_llm()
runnable = setup_runnable(model, system_prompt)

# Get user input
if prompt := st.chat_input("এখানে আপনার প্রশ্ন বা সমস্যা লিখুন..."):  # Enter your question or problem here...
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