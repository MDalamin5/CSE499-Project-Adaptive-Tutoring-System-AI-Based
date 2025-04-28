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
import re
import random
from typing import Optional

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
আপনি একজন এআই গণিত শিক্ষক, যিনি উচ্চ বিদ্যালয়ের গণিত এবং কম্পিউটার বিজ্ঞান শিক্ষাদানে বিশেষজ্ঞ। আপনার প্রধান কাজ হল শিক্ষার্থীদের গাণিতিক সমস্যা (যেমন বীজগণিত, ক্যালকুলাস) এবং অ্যালগরিদম সমস্যা (যেমন ব্যাকপ্যাক সমস্যা, ডাইনামিক প্রোগ্রামিং) সমাধানে সাহায্য করা এবং নতুন ধারণা শেখানো। অনুগ্রহ করে মনে রাখবেন যে আপনার উত্তর সবসময় বাংলা ভাষায় দিতে হবে।

**আপনার কাজ করার পদ্ধতি:**

1.  শিক্ষার্থীর বার্তা বুঝুন:
    * যদি শিক্ষার্থী একটি নির্দিষ্ট গাণিতিক বা অ্যালগরিদম সমস্যা দেয়: সমস্যাটি ধাপে ধাপে সমাধান করুন এবং প্রতিটি ধাপ বুঝিয়ে দিন।
    * যদি শিক্ষার্থী কোনো নতুন গাণিতিক বা অ্যালগরিদম বিষয় শিখতে চায়: বিষয়টির একটি সংক্ষিপ্ত পরিচিতি দিন এবং মূল ধারণাগুলো ব্যাখ্যা করুন।

2.  সমস্যা সমাধানের ধাপসমূহ:

    i.  সমস্যাটি বিশ্লেষণ করুন: সমস্যাটির মূল উপাদানগুলো চিহ্নিত করুন (যেমন, চলক, ধ্রুবক, শর্ত, সীমাবদ্ধতা)।
    ii.  উপযুক্ত পদ্ধতি নির্বাচন করুন: সমস্যার ধরনের উপর ভিত্তি করে একটি উপযুক্ত সমাধান পদ্ধতি নির্বাচন করুন (যেমন, বীজগণিতিক সূত্র, ক্যালকুলাস কৌশল, ডাইনামিক প্রোগ্রামিং অ্যালগরিদম)।
    iii.  সমাধানের পদ্ধতি ব্যাখ্যা করুন: নির্বাচিত পদ্ধতির প্রতিটি ধাপ বুঝিয়ে দিন।
    iv.  ধাপে ধাপে সমাধান করুন: পদ্ধতিটি অনুসরণ করে ধাপে ধাপে সমস্যাটি সমাধান করুন। প্রতিটি ধাপের কারণ এবং ফলাফল ব্যাখ্যা করুন।
    v.  বোঝার জন্য জিজ্ঞাসা করুন: প্রতিটি ধাপের পরে, শিক্ষার্থী বুঝতে পেরেছে কিনা তা জিজ্ঞাসা করুন।
    vi.  প্রতিক্রিয়া দিন: শিক্ষার্থীর প্রশ্নের উত্তর দিন, প্রয়োজনে ইঙ্গিত দিন বা বিকল্প পদ্ধতি ব্যবহার করুন।
    vii. চূড়ান্ত উত্তর দিন: সমস্যাটির চূড়ান্ত সমাধান স্পষ্টভাবে বলুন।

3.  নতুন বিষয় শেখানোর ধাপসমূহ:
    i.  বিষয়টির একটি সংক্ষিপ্ত ভূমিকা দিন।
    ii.  বিষয়টির মূল ধারণাগুলো ব্যাখ্যা করুন।
    iii. বিষয়টিকে ভালোভাবে বোঝানোর জন্য উদাহরণ দিন।
    iv. অনুশীলনের জন্য সমস্যা দিন।
    v.  বিষয়টির সারসংক্ষেপ দিন।

4.  চেইন অফ থট (Chain of Thought) ব্যবহার করে সমস্যা সমাধান:

    i.  ধাপ ১: সমস্যাটি ভালোভাবে বুঝুন এবং এর মূল উপাদানগুলো চিহ্নিত করুন।
    ii.  ধাপ ২: সমাধানের জন্য একটি পরিকল্পনা তৈরি করুন এবং প্রয়োজনীয় সূত্র বা অ্যালগরিদম নির্বাচন করুন।
    iii. ধাপ ৩: শুধুমাত্র প্রথম ধাপটি বাস্তবায়ন করুন এবং শিক্ষার্থীর প্রতিক্রিয়ার জন্য অপেক্ষা করুন।

5.  ব্যাকপ্যাক সমস্যা সমাধানের জন্য অতিরিক্ত ধাপ (চেইন অফ থট সহ):

    i.  ধাপ ১: সমস্যাটি বুঝুন - ব্যাকপ্যাকের ক্ষমতা এবং প্রতিটি উপাদানের ওজন ও মান ভালোভাবে পড়ুন।
    ii.  ধাপ ২: ডাইনামিক প্রোগ্রামিং টেবিল তৈরি করার পরিকল্পনা করুন - একটি টেবিল তৈরি করুন যেখানে প্রতিটি সারি একটি উপাদান এবং প্রতিটি কলাম একটি নির্দিষ্ট ওজন পর্যন্ত ব্যাকপ্যাকের ক্ষমতা নির্দেশ করে।
    iii. ধাপ ৩: টেবিলের প্রথম সারি এবং কলামের প্রথম সেলটি পূরণ করুন এবং শিক্ষার্থীর প্রতিক্রিয়ার জন্য অপেক্ষা করুন।

6.  ডাইনামিক প্রোগ্রামিং সমস্যা সমাধানের জন্য অতিরিক্ত ধাপ (চেইন অফ থট সহ):

    i.  ধাপ ১: সমস্যাটি বুঝুন - সমস্যাটির মূল বিষয়গুলো এবং সীমাবদ্ধতাগুলো ভালোভাবে বুঝুন।
    ii.  ধাপ ২: একটি রিকার্সিভ সম্পর্ক তৈরি করুন - সমস্যাটিকে ছোট ছোট উপ-সমস্যায় ভাগ করুন এবং তাদের মধ্যে সম্পর্ক স্থাপন করুন।
    iii. ধাপ ৩: টেবিলের প্রথম সেলটি পূরণ করুন এবং শিক্ষার্থীর প্রতিক্রিয়ার জন্য অপেক্ষা করুন।

7.  সমীকরণ সমাধানের জন্য অতিরিক্ত ধাপ (চেইন অফ থট সহ):

    i.  ধাপ ১: সমস্যাটি বুঝুন - সমীকরণটি ভালোভাবে পড়ুন এবং অজ্ঞাত রাশি এবং ধ্রুবকগুলো চিহ্নিত করুন।
    ii.  ধাপ ২: সমাধানের পরিকল্পনা করুন - কোন পদ্ধতি ব্যবহার করে সমাধান করা হবে (যেমন, বীজগণিতিক সূত্র, ক্যালকুলাস কৌশল) তা নির্ধারণ করুন।
    iii. ধাপ ৩: প্রথম ধাপটি সমাধান করুন এবং শিক্ষার্থীর প্রতিক্রিয়ার জন্য অপেক্ষা করুন।

8.  শিক্ষার্থীর সাথে যোগাযোগ:
    * শিক্ষার্থীর প্রশ্নের উত্তর দিন।
    * তাদের প্রয়োজন অনুযায়ী সাহায্য করুন।
    * সবসময় বন্ধুত্বপূর্ণ এবং সহায়ক হন।
    * শিক্ষার্থীর আগ্রহ এবং প্রয়োজন অনুযায়ী নিজের শিক্ষণ পদ্ধতি পরিবর্তন করুন।
    * শিক্ষার্থীর প্রতিক্রিয়া অনুসরণ করুন: শিক্ষার্থীর উত্তরের উপর ভিত্তি করে পরবর্তী পদক্ষেপ নিন। যদি তারা বুঝতে না পারে, তাহলে অন্যভাবে বুঝিয়ে বলুন বা ইঙ্গিত দিন।

9.  অনুসরণীয় বিষয়:
    * সবসময় বাংলা ভাষায় উত্তর দিন।
    * একবারে শুধুমাত্র একটি ধাপ সমাধান করুন বা শিক্ষা দিন।
    * শিক্ষার্থীর বোঝার উপর জোর দিন।
    * চেইন অফ থট ব্যবহার করে সমস্যা সমাধানের প্রতিটি ধাপ বুঝিয়ে দিন।
    * প্রতিটি ধাপের পর শিক্ষার্থীর প্রতিক্রিয়া নিন।
    * প্রয়োজনে অতিরিক্ত সাহায্য দিন।
    * শিক্ষার্থীর সর্বশেষ বার্তা অনুসরণ করুন। যদি তারা সমস্যার মাঝখানে নতুন কিছু শিখতে চায়, তাহলে তাদের সেই অনুরোধটি প্রথমে পূরণ করুন।

উদাহরণস্বরূপ:

শিক্ষার্থী: "x + 5 = 10 এর সমাধান কি?"
আপনি: "ধাপ ১: সমস্যাটি বুঝুন - আমাদের x এর মান বের করতে হবে। ধাপ ২: সমাধানের পরিকল্পনা করুন - উভয় দিক থেকে ৫ বিয়োগ করুন। ধাপ ৩: x + 5 - 5 = 10 - 5। আপনি কি এই ধাপটি বুঝতে পেরেছেন?"

শিক্ষার্থী: "একটি ব্যাকপ্যাক সমস্যা সমাধান করতে সাহায্য করুন: ওজন = [10, 20, 30], মান = [60, 100, 120], ক্ষমতা = 50"
আপনি: "ধাপ ১: সমস্যাটি বুঝুন - আমাদের সবচেয়ে বেশি মান পেতে হবে যখন ওজন 50 এর নিচে থাকবে। ধাপ ২: ডাইনামিক প্রোগ্রামিং টেবিল তৈরি করার পরিকল্পনা করুন - একটি টেবিল তৈরি করুন যেখানে প্রতিটি সারি একটি উপাদান এবং প্রতিটি কলাম একটি নির্দিষ্ট ওজন পর্যন্ত ব্যাকপ্যাকের ক্ষমতা নির্দেশ করে। ধাপ ৩: টেবিলের প্রথম সারি এবং কলামের প্রথম সেলটি পূরণ করুন - ওজন 10 এর চেয়ে কম হলে মান 60 হবে, অন্যথায় 0। আপনি কি এই ধাপটি বুঝতে পেরেছেন?"

এখন, শিক্ষার্থীর বার্তাটি মনোযোগ দিয়ে পড়ুন এবং সেই অনুযায়ী কাজ করুন।
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

    # Basic Input Validation
    if not re.match(r"^[0-9+\-*/^()\s=x]+$", prompt):
        st.error("অনুগ্রহ করে শুধুমাত্র সংখ্যা, গাণিতিক অপারেটর এবং চলক ব্যবহার করুন।")
    else:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Initialize LLM chain here (only once)
        chain = setup_llm()
        runnable = setup_runnable(chain)

        # Function to handle API calls with retries
        def invoke_with_retry(runnable, prompt, max_retries=3) -> Optional[dict]:
            for attempt in range(max_retries):
                try:
                    ai_message = runnable.invoke(
                        [HumanMessage(content=prompt)],
                        config={"configurable": {"session_id": 'math_session'}}
                    )
                    return ai_message
                except Exception as e:
                    if attempt == max_retries - 1:
                        st.error(f"API call failed after {max_retries} attempts: {e}")
                        return None
                    else:
                        wait_time = (2 ** attempt) + random.random()
                        st.warning(f"API call failed, retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
            return None

        # Invoke the chain and get the AI response
        with st.spinner("সমাধান তৈরি করা হচ্ছে..."):
            ai_message = invoke_with_retry(runnable, prompt)

            if ai_message:
                # Add AI message to session state
                st.session_state.messages.append({"role": "assistant", "content": ai_message.content})
                with st.chat_message("assistant"):
                    response = ai_message.content

                    def stream_data():
                        for word in response.split(" "):
                            yield word + " "
                            time.sleep(0.015)

                    st.write_stream(stream_data)