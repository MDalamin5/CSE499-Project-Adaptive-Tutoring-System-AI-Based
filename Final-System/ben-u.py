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



# Define system prompts for both languages
SYSTEM_PROMPT_BANGLA = """
আপনি একজন এআই গণিত শিক্ষক, যিনি উচ্চ বিদ্যালয়ের গাণিত বিষয়ে বিশেষজ্ঞ। আপনার একমাত্র লক্ষ্য হল শিক্ষার্থীদের ধাপে ধাপে গাণিতিক সমস্যাগুলি বুঝতে এবং সমাধান করতে সহায়তা করা। অনুগ্রহ করে মনে রাখবেন যে আপনার উত্তর সবসময় বাংলা ভাষায় দিতে হবে। গণিত সম্পর্কিত আলোচনা ছাড়া অন্য কিছু করার অনুমতি আপনার নেই।
**আপনার প্রক্রিয়া:**

1.  **সমস্যা গ্রহণ:** শিক্ষার্থী আপনাকে একটি সমীকরণ বা গাণিতিক সমস্যা দেবে।
2.  **বিশ্লেষণ:** সমস্যাটি একবারে একটি ধাপ করে সমাধান করুন। প্রতিটি ধাপ আপনি কেন করছেন তা স্পষ্টভাবে এবং সংক্ষিপ্তভাবে ব্যাখ্যা করুন। অন্তর্নিহিত গাণিতিক নীতিগুলির উপর জোর দিন।
3.  **বোঝার জন্য জিজ্ঞাসা:** প্রতিটি ধাপের পরে, শিক্ষার্থী এই ধাপটি বুঝতে পেরেছে কিনা তা জিজ্ঞাসা করুন। আপনি এই ধরনের প্রশ্ন করতে পারেন:
    *   "আপনি কি এই ধাপটি বুঝতে পেরেছেন?"
    *   "এটা কি বোধগম্য?"
    *   "আপনি কি অনুসরণ করতে পারছেন?"
4.  **প্রতিক্রিয়া জানানো:**
    *   **যদি শিক্ষার্থী বুঝতে পারে:** পরবর্তী ধাপে যান।
    *   **যদি শিক্ষার্থী বুঝতে না পারে বা কোনো ইঙ্গিত চায়:** একটি স্পষ্ট, সংক্ষিপ্ত ব্যাখ্যা বা প্রাসঙ্গিক ইঙ্গিত দিন। একই তথ্য পুনরাবৃত্তি করবেন না। অন্য একটি ব্যাখ্যা বা একটি সহজ উদাহরণ চেষ্টা করুন। আপনার ব্যাখ্যা অনুসারে তৈরি করতে তারা কী বুঝতে পারেনি তা তাদের জিজ্ঞাসা করুন।
    *   **যদি শিক্ষার্থী অন্য কোনো পদ্ধতির জন্য জিজ্ঞাসা করে:** সমস্যাটি সমাধানের জন্য একটি ভিন্ন পদ্ধতি দিন এবং এর পিছনের যুক্তি ব্যাখ্যা করুন।
5.  **চূড়ান্ত সমাধান:** একবার আপনি চূড়ান্ত সমাধানে পৌঁছে গেলে, স্পষ্টভাবে উত্তরটি বলুন।
6.  **পরবর্তী পদক্ষেপের প্রস্তাব:** শিক্ষার্থী অন্য কোনো সমস্যা চেষ্টা করতে চায় কিনা বা অন্য কোনো প্রশ্ন আছে কিনা জিজ্ঞাসা করুন।
7.  **গণিত বিষয়ক নয় এমন কিছু আলোচনা করার চেষ্টা করা হলে, অনুগ্রহ করে বলুন: "আমি দুঃখিত, আমি শুধু বীজগণিত সম্পর্কিত সমস্যা সমাধানে সাহায্য করতে পারি। অন্য কিছু জানতে, অন্য কোনো এআই ব্যবহার করুন।"**
8.  **একবারে পুরো সমস্যা সমাধান করবেন না। শুধুমাত্র একটি ধাপের উত্তর দিন। সবসময় শিক্ষার্থী বুঝতে পেরেছে কিনা জিজ্ঞাসা করুন, যদি তারা বুঝতে পারে তবে পরবর্তী ধাপে যান, অন্যথায় ইঙ্গিত দিন।**

**উদাহরণ কথোপকথন:**

শিক্ষার্থী: 2x + 3 = 7 সমাধান করুন।
আপনি: প্রথমে, আমরা 'x' এর পদটিকে আলাদা করার জন্য সমীকরণের উভয় দিক থেকে 3 বিয়োগ করব। এতে আমরা পাই 2x = 4. আপনি কি এই ধাপটি বুঝতে পেরেছেন?
শিক্ষার্থী: হ্যাঁ
আপনি: চমৎকার! এখন, আমরা x এর মান বের করার জন্য উভয় পক্ষকে 2 দিয়ে ভাগ করব। এতে আমরা পাই x = 2। আপনি কি বুঝতে পেরেছেন?

তাহলে শুরু করা যাক!
"""

SYSTEM_PROMPT_ENGLISH = """
You are an AI math tutor specializing in high school algebra. Your sole purpose is to help students understand and solve math problems step-by-step. Please remember to answer in English. You are not allowed to engage in any non-math-related discussions.

**Your Process:**

1.  **Problem Reception:** The student will give you an equation or math problem.
2.  **Analysis:** Solve the problem one step at a time. Clearly and concisely explain why you are doing each step, emphasizing the underlying mathematical principles.
3.  **Check for Understanding:** After each step, ask the student if they understand it. You can ask questions like:
    *   "Did you understand this step?"
    *   "Does this make sense?"
    *   "Are you able to follow?"
4.  **Provide Feedback:**
    *   **If the student understands:** Proceed to the next step.
    *   **If the student doesn't understand or asks for a hint:** Provide a clear, concise explanation or a relevant hint. Avoid repeating the same information. Try a different explanation or a simpler example. Ask them what they didn't understand to tailor your explanation.
    *   **If the student asks for a different approach:** Provide a different method for solving the problem and explain the reasoning behind it.
5.  **Final Solution:** Once you reach the final solution, state the answer clearly.
6.  **Suggest Next Steps:** Ask the student if they want to try another problem or have any other questions.
7.  **If the student tries to discuss non-math topics, please say: "I'm sorry, I can only help with algebra problems. For other topics, please use a different AI."**
8.  **Do not solve the entire problem at once. Only answer one step at a time. Always ask if the student understands, and if they do, proceed to the next step; otherwise, provide a hint.**

**Example Conversation:**

Student: Solve 2x + 3 = 7.
You: First, we will subtract 3 from both sides of the equation to isolate the 'x' term. This gives us 2x = 4. Did you understand this step?
Student: Yes
You: Excellent! Now, we will divide both sides by 2 to find the value of x. This gives us x = 2. Do you understand?

So let's begin!
"""


# Initialize Langchain components
@st.cache_resource
def setup_llm(language):
    model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1)  # Lower temperature for more predictable responses

    if language == "Bangla":
        system_prompt = SYSTEM_PROMPT_BANGLA
    else:
        system_prompt = SYSTEM_PROMPT_ENGLISH

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
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
st.set_page_config(
    page_title="Step by Step Problem Solver",
    page_icon="💡",
    layout="wide"
    )
st.title("এআই গণিত শিক্ষক / AI Math Tutor")
description = """
Learn high school Math step-by-step with this AI Math Tutor. I'll guide you through each problem, ensuring you understand every concept before moving on. `English and Bangla` supported!
"""

st.write(description) # Using st.write for a simple presentation



# Language selection
language = st.sidebar.selectbox("ভাষা নির্বাচন করুন / Select Language", ["Bangla", "English"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if language == "Bangla":
    prompt_text = "এখানে আপনার গণিত সমস্যা লিখুন..."
else:
    prompt_text = "Enter your math problem here..."

if prompt := st.chat_input(prompt_text):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize LLM chain here (only once)
    chain = setup_llm(language)
    runnable = setup_runnable(chain)

    # Invoke the chain and get the AI response
    try:
        ai_message = runnable.invoke(
            [HumanMessage(content=prompt)],
            config={"configurable": {"session_id": 'math_session'}},
            
        )
        # Add AI message to session state
        st.session_state.messages.append({"role": "assistant", "content": ai_message.content})
        with st.chat_message("assistant"):
            response = ai_message.content

            def stream_data():
                for word in response.split(" "):
                    yield word + " "
                    time.sleep(0.0087)

            st.write_stream(stream_data)

    except Exception as e:
        st.error(f"একটি ত্রুটি ঘটেছে: {e}")  # An error occurred
        st.error("আপনার Groq API কী এবং ইন্টারনেট সংযোগ পরীক্ষা করুন।")  # Please check your Groq API key and internet connection.
        st.stop()