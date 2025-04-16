import streamlit as st
import os
import re
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Check if API key is available
if not groq_api_key:
    st.error("Groq API key not found. Please configure it in the setup page.")
    st.stop()

# Define Pydantic models for structured output parsing
class StepSolution(BaseModel):
    step_explanation: str = Field(
        description="The explanation of the current step in solving the problem"
    )
    understanding_check: str = Field(
        description="A question to check if the student understands this step"
    )
    is_final_step: bool = Field(
        description="Whether this is the final step in solving the problem"
    )
    next_hint: str = Field(
        description="A hint about what the next step would be (if not final step)"
    )

# Custom output parser with robust JSON extraction
class RobustOutputParser:
    def __init__(self, pydantic_parser):
        self.pydantic_parser = pydantic_parser
        self.model_class = pydantic_parser.pydantic_object
    
    def parse(self, text):
        # Try to extract JSON from the text using regex
        json_match = re.search(r'({.*})', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                # Try to parse the extracted JSON
                parsed_json = json.loads(json_str)
                # Create an instance of the Pydantic model
                return self.model_class(**parsed_json)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to clean up the JSON string
                cleaned_json = self._clean_json_string(json_str)
                try:
                    parsed_json = json.loads(cleaned_json)
                    return self.model_class(**parsed_json)
                except:
                    # If all parsing attempts fail, create a default instance
                    if self.model_class == StepSolution:
                        return StepSolution(
                            step_explanation="I'll help you solve this step by step. Let's start by understanding the problem.",
                            understanding_check="Do you understand how we should approach this problem?",
                            is_final_step=False,
                            next_hint="We'll need to identify the key mathematical concepts involved."
                        )
        else:
            # If no JSON-like structure is found, try to extract information from the text
            try:
                # For non-JSON responses, try to intelligently extract the parts
                paragraphs = text.split('\n\n')
                explanation = paragraphs[0] if paragraphs else text
                check = "Do you understand this step?" 
                for p in paragraphs:
                    if "?" in p and any(word in p.lower() for word in ["understand", "sense", "follow", "clear"]):
                        check = p
                        break
                
                is_final = "final" in text.lower() or "answer" in text.lower() or "solution" in text.lower()
                next_hint = ""
                for p in paragraphs:
                    if "next" in p.lower() and not is_final:
                        next_hint = p
                        break
                
                return StepSolution(
                    step_explanation=explanation,
                    understanding_check=check,
                    is_final_step=is_final,
                    next_hint=next_hint if not is_final else ""
                )
            except:
                # If all extraction attempts fail, create a default instance
                return StepSolution(
                    step_explanation="I'll help you solve this step by step. Let's start by understanding the problem.",
                    understanding_check="Do you understand how we should approach this problem?",
                    is_final_step=False,
                    next_hint="We'll need to identify the key mathematical concepts involved."
                )
    
    def _clean_json_string(self, json_str):
        # Remove any thinking tags
        json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
        # Remove any non-JSON text before the first {
        json_str = re.sub(r'^[^{]*', '', json_str)
        # Remove any non-JSON text after the last }
        json_str = re.sub(r'}[^}]*$', '}', json_str)
        return json_str

# Initialize parsers
step_solution_parser = PydanticOutputParser(pydantic_object=StepSolution)
robust_step_solution_parser = RobustOutputParser(step_solution_parser)

# Define system prompts for both languages
SYSTEM_PROMPT_BANGLA = """
আপনি একজন এআই গণিত শিক্ষক, যিনি উচ্চ বিদ্যালয়ের বীজগণিত বিষয়ে বিশেষজ্ঞ। আপনার লক্ষ্য হল শিক্ষার্থীদের ধাপে ধাপে গাণিতিক সমস্যাগুলি বুঝতে এবং সমাধান করতে সহায়তা করা। অনুগ্রহ করে মনে রাখবেন যে আপনার উত্তর সবসময় বাংলা ভাষায় দিতে হবে।

**আপনার প্রক্রিয়া:**

1. **সমস্যা গ্রহণ:** শিক্ষার্থী আপনাকে একটি সমীকরণ বা গাণিতিক সমস্যা দেবে।
2. **বিশ্লেষণ:** সমস্যাটি একবারে একটি ধাপ করে সমাধান করুন। প্রতিটি ধাপ আপনি কেন করছেন তা স্পষ্টভাবে এবং সংক্ষিপ্তভাবে ব্যাখ্যা করুন। অন্তর্নিহিত গাণিতিক নীতিগুলির উপর জোর দিন।
3. **বোঝার জন্য জিজ্ঞাসা:** প্রতিটি ধাপের পরে, শিক্ষার্থী এই ধাপটি বুঝতে পেরেছে কিনা তা জিজ্ঞাসা করুন।
4. **প্রতিক্রিয়া জানানো:**
   * **যদি শিক্ষার্থী বুঝতে পারে:** পরবর্তী ধাপে যান।
   * **যদি শিক্ষার্থী বুঝতে না পারে বা কোনো ইঙ্গিত চায়:** একটি স্পষ্ট, সংক্ষিপ্ত ব্যাখ্যা বা প্রাসঙ্গিক ইঙ্গিত দিন।
5. **চূড়ান্ত সমাধান:** একবার আপনি চূড়ান্ত সমাধানে পৌঁছে গেলে, স্পষ্টভাবে উত্তরটি বলুন।

**গুরুত্বপূর্ণ নির্দেশনা:**
- একবারে পুরো সমস্যা সমাধান করবেন না। শুধুমাত্র একটি ধাপের উত্তর দিন।
- সবসময় শিক্ষার্থী বুঝতে পেরেছে কিনা জিজ্ঞাসা করুন।
- আপনার উত্তর অবশ্যই নিম্নলিখিত JSON ফরম্যাটে হতে হবে:

```json
{
  "step_explanation": "এই ধাপে আমরা যা করছি তার ব্যাখ্যা",
  "understanding_check": "আপনি কি এই ধাপটি বুঝতে পেরেছেন?",
  "is_final_step": false,
  "next_hint": "পরবর্তী ধাপে আমরা কী করব তার ইঙ্গিত"
}
```

যদি এটি চূড়ান্ত ধাপ হয়, তাহলে `is_final_step` কে `true` করুন এবং `next_hint` খালি রাখুন।

**উদাহরণ:**

শিক্ষার্থী: 2x + 3 = 7 সমাধান করুন।
আপনি:
```json
{
  "step_explanation": "প্রথমে, আমরা 'x' এর পদটিকে আলাদা করার জন্য সমীকরণের উভয় দিক থেকে 3 বিয়োগ করব। এতে আমরা পাই 2x = 4।",
  "understanding_check": "আপনি কি এই ধাপটি বুঝতে পেরেছেন?",
  "is_final_step": false,
  "next_hint": "পরবর্তী ধাপে আমরা x এর মান বের করার জন্য উভয় পক্ষকে 2 দিয়ে ভাগ করব।"
}
```

শিক্ষার্থী: হ্যাঁ
আপনি:
```json
{
  "step_explanation": "চমৎকার! এখন, আমরা x এর মান বের করার জন্য উভয় পক্ষকে 2 দিয়ে ভাগ করব। এতে আমরা পাই x = 2।",
  "understanding_check": "আপনি কি বুঝতে পেরেছেন?",
  "is_final_step": true,
  "next_hint": ""
}
```

তাহলে শুরু করা যাক!
"""

SYSTEM_PROMPT_ENGLISH = """
You are an AI math tutor specializing in high school algebra. Your goal is to help students understand and solve math problems step-by-step. Please remember to answer in English.

**Your Process:**

1. **Problem Reception:** The student will give you an equation or math problem.
2. **Analysis:** Solve the problem one step at a time. Clearly and concisely explain why you are doing each step, emphasizing the underlying mathematical principles.
3. **Check for Understanding:** After each step, ask the student if they understand it.
4. **Provide Feedback:**
   * **If the student understands:** Proceed to the next step.
   * **If the student doesn't understand or asks for a hint:** Provide a clear, concise explanation or a relevant hint.
5. **Final Solution:** Once you reach the final solution, state the answer clearly.

**Important Instructions:**
- Do not solve the entire problem at once. Only answer one step at a time.
- Always ask if the student understands.
- Your response must be in the following JSON format:

```json
{
  "step_explanation": "Explanation of what we're doing in this step",
  "understanding_check": "Did you understand this step?",
  "is_final_step": false,
  "next_hint": "Hint about what we'll do in the next step"
}
```

If this is the final step, set `is_final_step` to `true` and leave `next_hint` empty.

**Example:**

Student: Solve 2x + 3 = 7.
You:
```json
{
  "step_explanation": "First, we will subtract 3 from both sides of the equation to isolate the 'x' term. This gives us 2x = 4.",
  "understanding_check": "Did you understand this step?",
  "is_final_step": false,
  "next_hint": "In the next step, we'll divide both sides by 2 to find the value of x."
}
```

Student: Yes
You:
```json
{
  "step_explanation": "Excellent! Now, we will divide both sides by 2 to find the value of x. This gives us x = 2.",
  "understanding_check": "Do you understand?",
  "is_final_step": true,
  "next_hint": ""
}
```

So let's begin!
"""

# Initialize Langchain components
@st.cache_resource
def setup_llm(language):
    # Initialize the base model
    model = ChatGroq(model_name="qwen-qwq-32b", temperature=0.1)
    
    # Select the appropriate system prompt based on language
    if language == "Bangla":
        system_prompt = SYSTEM_PROMPT_BANGLA
    else:
        system_prompt = SYSTEM_PROMPT_ENGLISH
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    
    # Create the chain
    chain = prompt | model
    return chain, model

# Setup the runnable with message history
@st.cache_resource(show_spinner=False)
def setup_runnable(_chain):
    store = {}
    
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    runnable = RunnableWithMessageHistory(_chain, get_session_history)
    return runnable

# Function to process model output for step solution
def process_step_solution_output(output_text, language):
    try:
        step_solution = robust_step_solution_parser.parse(output_text)
        return step_solution
    except Exception as e:
        st.sidebar.error(f"Error parsing step solution: {str(e)}")
        # Create a default response based on the language
        if language == "Bangla":
            return StepSolution(
                step_explanation="আমি আপনাকে ধাপে ধাপে এই সমস্যা সমাধান করতে সাহায্য করব।",
                understanding_check="আপনি কি এই পদ্ধতি বুঝতে পেরেছেন?",
                is_final_step=False,
                next_hint="আমরা প্রথমে সমস্যাটি ভালভাবে বুঝতে চেষ্টা করব।"
            )
        else:
            return StepSolution(
                step_explanation="I'll help you solve this problem step by step.",
                understanding_check="Do you understand this approach?",
                is_final_step=False,
                next_hint="We'll first try to understand the problem clearly."
            )

# Main Streamlit UI
def main():
    st.title("Step-by-Step Math Problem Solver")
    
    # Language selection
    language = st.sidebar.selectbox(
        "ভাষা নির্বাচন করুন / Select Language", 
        ["English", "Bangla"]
    )
    
    # Display appropriate header based on language
    if language == "Bangla":
        st.header("ধাপে ধাপে গণিত সমস্যা সমাধানকারী")
        st.markdown("এই টুলটি আপনাকে ধাপে ধাপে গণিত সমস্যা সমাধান করতে সাহায্য করবে। প্রতিটি ধাপের পরে, আপনি বুঝতে পেরেছেন কিনা তা জানাতে হবে।")
    else:
        st.header("Step-by-Step Math Problem Solver")
        st.markdown("This tool will help you solve math problems step by step. After each step, you'll need to indicate whether you understand.")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    
    if "problem_solved" not in st.session_state:
        st.session_state.problem_solved = False
    
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "step_data" in message:
                step_data = message["step_data"]
                
                # Display the step explanation
                st.markdown(step_data["step_explanation"])
                
                # Display the understanding check
                st.markdown(f"**{step_data['understanding_check']}**")
                
                # If it's the final step, add a visual indicator
                if step_data["is_final_step"]:
                    if language == "Bangla":
                        st.success("সমাধান সম্পূর্ণ হয়েছে! ✅")
                    else:
                        st.success("Solution completed! ✅")
            else:
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
        
        try:
            # Initialize LLM components
            chain, model = setup_llm(language)
            runnable = setup_runnable(chain)
            
            # Invoke the chain with the updated prompt
            ai_message = runnable.invoke(
                [HumanMessage(content=prompt)],
                config={"configurable": {"session_id": 'math_session'}}
            )
            
            # Process the response to extract structured data
            step_solution = process_step_solution_output(ai_message.content, language)
            
            # Add AI message to session state with structured format
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_message.content,  # For backward compatibility
                "step_data": {
                    "step_explanation": step_solution.step_explanation,
                    "understanding_check": step_solution.understanding_check,
                    "is_final_step": step_solution.is_final_step,
                    "next_hint": step_solution.next_hint
                }
            })
            
            # Display the message with streaming effect
            with st.chat_message("assistant"):
                # Display the step explanation with streaming effect
                response = step_solution.step_explanation
                
                def stream_data():
                    for word in response.split(" "):
                        yield word + " "
                        time.sleep(0.015)
                
                st.write_stream(stream_data)
                
                # Display the understanding check
                st.markdown(f"**{step_solution.understanding_check}**")
                
                # If it's the final step, add a visual indicator
                if step_solution.is_final_step:
                    if language == "Bangla":
                        st.success("সমাধান সম্পূর্ণ হয়েছে! ✅")
                    else:
                        st.success("Solution completed! ✅")
                
            # Update the current step
            st.session_state.current_step += 1
            
            # Update problem solved status
            if step_solution.is_final_step:
                st.session_state.problem_solved = True
                
        except Exception as e:
            if language == "Bangla":
                st.error(f"একটি ত্রুটি ঘটেছে: {e}")
                st.error("আপনার Groq API কী এবং ইন্টারনেট সংযোগ পরীক্ষা করুন।")
            else:
                st.error(f"An error occurred: {e}")
                st.error("Please check your Groq API key and internet connection.")
            st.stop()
    
    # Add a reset button
    st.sidebar.divider()
    if st.sidebar.button("Reset / রিসেট"):
        st.session_state.messages = []
        st.session_state.current_step = 0
        st.session_state.problem_solved = False
        st.rerun()

if __name__ == "__main__":
    main()
