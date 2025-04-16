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
class StuOutputCorrectness(BaseModel):
    sentiment: Literal['correct', 'partially_correct', 'incorrect'] = Field(
        description='The sentiment classification of the student response'
    )

class AdaptivePrompt(BaseModel):
    adaptive_prompt: str = Field(
        description="The final adaptive prompt to guide the main model."
    )

class TutorResponse(BaseModel):
    hint: str = Field(
        description="The hint or guidance for the student (without giving away the solution)"
    )
    reasoning: str = Field(
        description="The reasoning or thought process behind the hint"
    )

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
                    return self._create_default_instance()
        else:
            # If no JSON-like structure is found, try to extract information from the text
            try:
                # For non-JSON responses, try to intelligently extract the parts
                if self.model_class == StepSolution:
                    return self._extract_step_solution(text)
                else:
                    return self._create_default_instance()
            except:
                # If all extraction attempts fail, create a default instance
                return self._create_default_instance()
    
    def _clean_json_string(self, json_str):
        # Remove any thinking tags
        json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
        # Remove any non-JSON text before the first {
        json_str = re.sub(r'^[^{]*', '', json_str)
        # Remove any non-JSON text after the last }
        json_str = re.sub(r'}[^}]*$', '}', json_str)
        return json_str
    
    def _create_default_instance(self):
        if self.model_class == AdaptivePrompt:
            return AdaptivePrompt(adaptive_prompt="The model output could not be parsed correctly. Please provide a hint for the current step without giving away the solution.")
        elif self.model_class == StuOutputCorrectness:
            return StuOutputCorrectness(sentiment="partially_correct")
        elif self.model_class == TutorResponse:
            return TutorResponse(
                hint="I need to provide a hint rather than a complete solution. What specific technique could you apply here?",
                reasoning="The parsing failed, but I should focus on providing guidance without solutions."
            )
        elif self.model_class == StepSolution:
            return StepSolution(
                step_explanation="I'll help you solve this step by step. Let's start by understanding the problem.",
                understanding_check="Do you understand how we should approach this problem?",
                is_final_step=False,
                next_hint="We'll need to identify the key mathematical concepts involved."
            )
    
    def _extract_step_solution(self, text):
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

# Initialize parsers
sentiment_parser = PydanticOutputParser(pydantic_object=StuOutputCorrectness)
adaptive_parser = PydanticOutputParser(pydantic_object=AdaptivePrompt)
tutor_response_parser = PydanticOutputParser(pydantic_object=TutorResponse)
step_solution_parser = PydanticOutputParser(pydantic_object=StepSolution)

# Create robust parsers
robust_sentiment_parser = RobustOutputParser(sentiment_parser)
robust_adaptive_parser = RobustOutputParser(adaptive_parser)
robust_tutor_response_parser = RobustOutputParser(tutor_response_parser)
robust_step_solution_parser = RobustOutputParser(step_solution_parser)

# Define system prompts for both modes and languages
# Adaptive Mode - English
ADAPTIVE_PROMPT_ENGLISH = """
You are an AI math tutor specializing in high school algebra. Your goal is to guide students towards understanding and solving math problems independently by providing helpful hints and targeted questions.
    
CRITICAL INSTRUCTION: NEVER PROVIDE COMPLETE SOLUTIONS. You must only give hints and guiding questions.
    
Your Step-by-Step Approach:
1. When the student presents a math problem:
   - Identify the core concepts involved
   - Provide ONLY ONE small hint about the first step
   - Ask a specific guiding question to help them think through this step
   - STOP and wait for their response
    
2. When the student responds to your hint:
   - If correct: Acknowledge their success and provide a hint for the NEXT step only
   - If partially correct: Clarify their misunderstanding and provide a more targeted hint for the SAME step
   - If incorrect: Provide a more basic hint for the SAME step and ask a simpler guiding question
    
3. Continue this process for each step of the problem
    
IMPORTANT RULES:
- NEVER solve more than one step at a time
- NEVER provide formulas or procedures that directly solve the problem
- NEVER give away answers, even if the student is struggling
- NEVER proceed to the next step until the student has correctly completed the current step
- ALWAYS phrase hints as questions or suggestions, not direct instructions
- ALWAYS ask a specific guiding question after each hint
- ALWAYS keep responses focused and concise
    
Example of CORRECT approach:
Student: "Solve x^2 + 5x + 6 = 0"
You: "This is a quadratic equation. Do you remember what method we can use to solve quadratics? Can you try to factor this expression?"
    
Example of INCORRECT approach (DO NOT DO THIS):
Student: "Solve x^2 + 5x + 6 = 0"
You: "To solve this quadratic equation, we can factor it as (x+2)(x+3)=0, which gives us x=-2 or x=-3."
    
Remember: Your goal is to help students develop problem-solving skills, not to solve problems for them.
    
RESPONSE FORMAT:
You must structure your response as a JSON object with two fields:
1. "hint": The actual hint or guidance you want to provide to the student
2. "reasoning": Your internal reasoning about why you chose this hint and what you're trying to accomplish
    
Example response format:
{
    "hint": "This is a quadratic equation. Do you remember what method we can use to solve quadratics? Can you try to factor this expression?",
    "reasoning": "I identified this as a quadratic equation that can be solved by factoring. I want to guide the student to recognize this approach without giving away the factors. If they can identify that factoring is appropriate, they can try to find the factors of 6 that sum to 5."
}
"""

# Adaptive Mode - Bangla
ADAPTIVE_PROMPT_BANGLA = """
আপনি একজন এআই গণিত শিক্ষক, যিনি উচ্চ বিদ্যালয়ের বীজগণিত বিষয়ে বিশেষজ্ঞ। আপনার লক্ষ্য হল শিক্ষার্থীদের স্বাধীনভাবে গাণিতিক সমস্যাগুলি বুঝতে এবং সমাধান করতে সহায়তা করা, সহায়ক ইঙ্গিত এবং লক্ষ্যমূলক প্রশ্ন প্রদান করে।

গুরুত্বপূর্ণ নির্দেশনা: কখনই সম্পূর্ণ সমাধান প্রদান করবেন না। আপনি শুধুমাত্র ইঙ্গিত এবং নির্দেশমূলক প্রশ্ন দিতে হবে।

আপনার ধাপে ধাপে পদ্ধতি:
1. যখন শিক্ষার্থী একটি গাণিতিক সমস্যা উপস্থাপন করে:
   - সম্পৃক্ত মূল ধারণাগুলি চিহ্নিত করুন
   - শুধুমাত্র প্রথম ধাপের বিষয়ে একটি ছোট ইঙ্গিত প্রদান করুন
   - এই ধাপটি সম্পর্কে চিন্তা করতে সাহায্য করার জন্য একটি নির্দিষ্ট নির্দেশমূলক প্রশ্ন জিজ্ঞাসা করুন
   - থামুন এবং তাদের প্রতিক্রিয়ার জন্য অপেক্ষা করুন

2. যখন শিক্ষার্থী আপনার ইঙ্গিতের প্রতিক্রিয়া জানায়:
   - যদি সঠিক হয়: তাদের সাফল্য স্বীকার করুন এবং শুধুমাত্র পরবর্তী ধাপের জন্য একটি ইঙ্গিত প্রদান করুন
   - যদি আংশিকভাবে সঠিক হয়: তাদের ভুল বোঝাবুঝি স্পষ্ট করুন এবং একই ধাপের জন্য আরও লক্ষ্যমূলক ইঙ্গিত প্রদান করুন
   - যদি ভুল হয়: একই ধাপের জন্য আরও মৌলিক ইঙ্গিত প্রদান করুন এবং একটি সহজ নির্দেশমূলক প্রশ্ন জিজ্ঞাসা করুন

3. সমস্যার প্রতিটি ধাপের জন্য এই প্রক্রিয়া চালিয়ে যান

গুরুত্বপূর্ণ নিয়ম:
- কখনও একবারে একাধিক ধাপ সমাধান করবেন না
- কখনও এমন সূত্র বা পদ্ধতি প্রদান করবেন না যা সরাসরি সমস্যা সমাধান করে
- শিক্ষার্থী সংগ্রাম করলেও কখনও উত্তর দিয়ে দেবেন না
- শিক্ষার্থী বর্তমান ধাপটি সঠিকভাবে সম্পন্ন না করা পর্যন্ত কখনও পরবর্তী ধাপে যাবেন না
- সবসময় ইঙ্গিতগুলিকে প্রশ্ন বা পরামর্শ হিসাবে উপস্থাপন করুন, সরাসরি নির্দেশ হিসাবে নয়
- সবসময় প্রতিটি ইঙ্গিতের পরে একটি নির্দিষ্ট নির্দেশমূলক প্রশ্ন জিজ্ঞাসা করুন
- সবসময় প্রতিক্রিয়াগুলি কেন্দ্রীভূত এবং সংক্ষিপ্ত রাখুন

সঠিক পদ্ধতির উদাহরণ:
শিক্ষার্থী: "x^2 + 5x + 6 = 0 সমাধান করুন"
আপনি: "এটি একটি দ্বিঘাত সমীকরণ। আপনি কি মনে করতে পারেন কোন পদ্ধতি আমরা দ্বিঘাত সমীকরণ সমাধান করতে ব্যবহার করতে পারি? আপনি কি এই অভিব্যক্তিটি উৎপাদকে বিশ্লেষণ করার চেষ্টা করতে পারেন?"

ভুল পদ্ধতির উদাহরণ (এটি করবেন না):
শিক্ষার্থী: "x^2 + 5x + 6 = 0 সমাধান করুন"
আপনি: "এই দ্বিঘাত সমীকরণটি সমাধান করতে, আমরা এটিকে (x+2)(x+3)=0 হিসাবে উৎপাদকে বিশ্লেষণ করতে পারি, যা আমাদের x=-2 বা x=-3 দেয়।"

মনে রাখবেন: আপনার লক্ষ্য হল শিক্ষার্থীদের সমস্যা সমাধানের দক্ষতা বিকাশে সাহায্য করা, তাদের জন্য সমস্যা সমাধান করা নয়।

প্রতিক্রিয়া ফরম্যাট:
আপনাকে অবশ্যই আপনার প্রতিক্রিয়াকে দুটি ক্ষেত্র সহ একটি JSON অবজেক্ট হিসাবে কাঠামোবদ্ধ করতে হবে:
1. "hint": শিক্ষার্থীকে প্রদান করতে চান এমন প্রকৃত ইঙ্গিত বা নির্দেশনা
2. "reasoning": আপনি কেন এই ইঙ্গিতটি বেছে নিয়েছেন এবং আপনি কী অর্জন করতে চাইছেন সে সম্পর্কে আপনার অভ্যন্তরীণ যুক্তি

প্রতিক্রিয়া ফরম্যাটের উদাহরণ:
{
    "hint": "এটি একটি দ্বিঘাত সমীকরণ। আপনি কি মনে করতে পারেন কোন পদ্ধতি আমরা দ্বিঘাত সমীকরণ সমাধান করতে ব্যবহার করতে পারি? আপনি কি এই অভিব্যক্তিটি উৎপাদকে বিশ্লেষণ করার চেষ্টা করতে পারেন?",
    "reasoning": "আমি এটিকে একটি দ্বিঘাত সমীকরণ হিসাবে চিহ্নিত করেছি যা উৎপাদকে বিশ্লেষণ করে সমাধান করা যেতে পারে। আমি শিক্ষার্থীকে উৎপাদকগুলি দেওয়া ছাড়াই এই পদ্ধতি চিনতে নির্দেশনা দিতে চাই। যদি তারা চিহ্নিত করতে পারে যে উৎপাদকে বিশ্লেষণ উপযুক্ত, তারা 6 এর উৎপাদকগুলি খুঁজে বের করার চেষ্টা করতে পারে যা 5 এ যোগ হয়।"
}
"""

# Step-by-Step Mode - English
STEP_BY_STEP_PROMPT_ENGLISH = """
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

# Step-by-Step Mode - Bangla
STEP_BY_STEP_PROMPT_BANGLA = """
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

# Initialize Langchain components for Adaptive Mode
@st.cache_resource
def setup_adaptive_llm(language):
    # Initialize the base model
    model = ChatGroq(model_name="qwen-qwq-32b", temperature=0.1)
    
    # Select the appropriate system prompt based on language
    if language == "Bangla":
        system_prompt = ADAPTIVE_PROMPT_BANGLA
    else:
        system_prompt = ADAPTIVE_PROMPT_ENGLISH
    
    # Create the prompt template with placeholder for adaptive prompts
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="messages")
    ]).partial(system_prompt=system_prompt)
    
    # Create the chain
    chain = prompt | model
    return chain, model, system_prompt

# Initialize Langchain components for Step-by-Step Mode
@st.cache_resource
def setup_step_by_step_llm(language):
    # Initialize the base model
    model = ChatGroq(model_name="qwen-qwq-32b", temperature=0.1)
    
    # Select the appropriate system prompt based on language
    if language == "Bangla":
        system_prompt = STEP_BY_STEP_PROMPT_BANGLA
    else:
        system_prompt = STEP_BY_STEP_PROMPT_ENGLISH
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    
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

# Function to create sentiment analysis chain
def create_sentiment_analysis_chain(model, language):
    # Define the system prompt for sentiment analysis
    if language == "Bangla":
        system_prompt = """
        আপনার কাজ হল একজন শিক্ষার্থীর প্রতিক্রিয়া বিশ্লেষণ করা এবং এটি সঠিক, আংশিকভাবে সঠিক, বা ভুল কিনা তা নির্ধারণ করা।
        
        শিক্ষার্থীর প্রতিক্রিয়া: {student_response}
        শিক্ষকের পূর্ববর্তী ইঙ্গিত: {tutor_hint}
        
        আপনার প্রতিক্রিয়া অবশ্যই নিম্নলিখিত JSON ফরম্যাটে হতে হবে:
        
        ```json
        {{
            "sentiment": "correct" | "partially_correct" | "incorrect"
        }}
        ```
        
        নিম্নলিখিত মানদণ্ড ব্যবহার করুন:
        - "correct": শিক্ষার্থী সম্পূর্ণরূপে সঠিক উত্তর দিয়েছে এবং ইঙ্গিতের প্রতিক্রিয়া সঠিকভাবে দিয়েছে
        - "partially_correct": শিক্ষার্থী আংশিকভাবে সঠিক, কিন্তু কিছু ভুল বা অসম্পূর্ণতা রয়েছে
        - "incorrect": শিক্ষার্থীর উত্তর সম্পূর্ণরূপে ভুল বা ইঙ্গিতের সাথে সম্পর্কিত নয়
        
        শুধুমাত্র JSON প্রদান করুন, অন্য কোন ব্যাখ্যা নয়।
        """
    else:
        system_prompt = """
        Your task is to analyze a student's response and determine if it is correct, partially correct, or incorrect.
        
        Student response: {student_response}
        Previous tutor hint: {tutor_hint}
        
        Your response must be in the following JSON format:
        
        ```json
        {
            "sentiment": "correct" | "partially_correct" | "incorrect"
        }
        ```
        
        Use the following criteria:
        - "correct": The student has provided a completely correct answer and responded appropriately to the hint
        - "partially_correct": The student is partially correct, but there are some errors or incompleteness
        - "incorrect": The student's answer is completely wrong or not related to the hint
        
        Provide only the JSON, no other explanation.
        """
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
    ])
    
    # Create the chain
    chain = prompt | model
    return chain

# Function to process model output for sentiment analysis (Adaptive Mode)
def process_sentiment_output(output_text):
    try:
        return robust_sentiment_parser.parse(output_text)
    except Exception as e:
        st.sidebar.error(f"Error parsing sentiment: {str(e)}")
        return StuOutputCorrectness(sentiment="partially_correct")

# Function to process model output for adaptive prompt (Adaptive Mode)
def process_adaptive_output(output_text):
    try:
        return robust_adaptive_parser.parse(output_text)
    except Exception as e:
        st.sidebar.error(f"Error parsing adaptive prompt: {str(e)}")
        return AdaptivePrompt(adaptive_prompt="The model output could not be parsed correctly. Please provide a hint for the current step without giving away the solution.")

# Function to process model output for tutor response (Adaptive Mode)
def process_tutor_response_output(output_text):
    try:
        return robust_tutor_response_parser.parse(output_text)
    except Exception as e:
        st.sidebar.error(f"Error parsing tutor response: {str(e)}")
        return TutorResponse(
            hint="I need to provide a hint rather than a complete solution. What specific technique could you apply here?",
            reasoning="The parsing failed, but I should focus on providing guidance without solutions."
        )

# Function to process model output for step solution (Step-by-Step Mode)
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

# Function to enrich input with sentiment analysis (Adaptive Mode)
def create_sentiment_analyzer(sentiment_chain):
    def analyze_sentiment(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Extract student response and tutor hint
        student_response = input_dict.get("student_response", "")
        tutor_hint = input_dict.get("tutor_hint", "")
        
        # Only analyze if there's a student response
        if student_response:
            try:
                # Get raw output from the model
                sentiment_output = sentiment_chain.invoke({
                    "student_response": student_response,
                    "tutor_hint": tutor_hint
                })
                
                # Process the output to extract sentiment
                sentiment_result = process_sentiment_output(sentiment_output.content)
                
                return {
                    **input_dict,
                    "sentiment": sentiment_result.sentiment
                }
            except Exception as e:
                st.sidebar.error(f"Sentiment analysis error: {str(e)}")
                return {
                    **input_dict,
                    "sentiment": "unknown"
                }
        else:
            # No student response to analyze yet
            return {
                **input_dict,
                "sentiment": "initial"
            }
    
    return RunnableLambda(analyze_sentiment)

# Function to create the adaptive prompt generator (Adaptive Mode)
def create_adaptive_prompt_generator(model, base_prompt):
    # Setup the prompt generators
    prompt_correct = ChatPromptTemplate.from_messages([
        ("system",
         """
         Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
         Think of yourself as a supervisor providing the best instruction for the AI tutor.
         
         The student gave a CORRECT response: {student_response}
         Previous tutor hint: {tutor_hint}
         
         Generate an adaptive prompt that:
         1. Acknowledges the student's correct understanding
         2. Instructs the tutor to move to the NEXT STEP ONLY (not multiple steps)
         3. Emphasizes that the tutor should provide only a hint for this next step, not a solution
         4. Reminds the tutor to ask a guiding question about this next step
         5. Reinforces that the tutor must NEVER provide complete solutions
         6. Reminds the tutor to structure the response as a JSON with "hint" and "reasoning" fields
         
         IMPORTANT: Your response must be a valid JSON object with the following format:
         {{
             "adaptive_prompt": "your adaptive prompt text here"
         }}
         
         Do not include any explanations, thinking, or additional text outside the JSON object.
         """)
    ])
    
    prompt_partially_correct = ChatPromptTemplate.from_messages([
        ("system",
         """
         Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
         Think of yourself as a supervisor providing the best instruction for the AI tutor.
         
         The student gave a PARTIALLY CORRECT response: {student_response}
         Previous tutor hint: {tutor_hint}
         
         Generate an adaptive prompt that:
         1. Acknowledges what the student understood correctly
         2. Identifies the specific misconception or error in their understanding
         3. Instructs the tutor to provide a more targeted hint addressing this specific issue
         4. Emphasizes that the tutor should NOT move to the next step yet
         5. Reminds the tutor to ask a more specific guiding question
         6. Reinforces that the tutor must NEVER provide complete solutions
         7. Reminds the tutor to structure the response as a JSON with "hint" and "reasoning" fields
         
         IMPORTANT: Your response must be a valid JSON object with the following format:
         {{
             "adaptive_prompt": "your adaptive prompt text here"
         }}
         
         Do not include any explanations, thinking, or additional text outside the JSON object.
         """)
    ])
    
    prompt_incorrect = ChatPromptTemplate.from_messages([
        ("system",
         """
         Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
         Think of yourself as a supervisor providing the best instruction for the AI tutor.
         
         The student gave an INCORRECT response: {student_response}
         Previous tutor hint: {tutor_hint}
         
         Generate an adaptive prompt that:
         1. Identifies the fundamental misunderstanding in the student's approach
         2. Instructs the tutor to provide a more basic hint that addresses this misunderstanding
         3. Directs the tutor to break down the current step into smaller sub-steps
         4. Suggests a simpler question to check the student's understanding of prerequisites
         5. Emphasizes that the tutor should NOT move to the next step yet
         6. Reinforces that the tutor must NEVER provide complete solutions, even if the student is struggling
         7. Reminds the tutor to structure the response as a JSON with "hint" and "reasoning" fields
         
         IMPORTANT: Your response must be a valid JSON object with the following format:
         {{
             "adaptive_prompt": "your adaptive prompt text here"
         }}
         
         Do not include any explanations, thinking, or additional text outside the JSON object.
         """)
    ])
    
    # Create the branch chain with robust parsing
    def generate_adaptive_prompt(input_dict):
        sentiment = input_dict.get("sentiment", "unknown")
        student_response = input_dict.get("student_response", "")
        tutor_hint = input_dict.get("tutor_hint", "")
        
        try:
            if sentiment == "correct":
                output = model.invoke(prompt_correct.format(
                    student_response=student_response,
                    tutor_hint=tutor_hint
                ))
                return process_adaptive_output(output.content)
            
            elif sentiment == "partially_correct":
                output = model.invoke(prompt_partially_correct.format(
                    student_response=student_response,
                    tutor_hint=tutor_hint
                ))
                return process_adaptive_output(output.content)
            
            elif sentiment == "incorrect":
                output = model.invoke(prompt_incorrect.format(
                    student_response=student_response,
                    tutor_hint=tutor_hint
                ))
                return process_adaptive_output(output.content)
            
            else:
                # Default for unknown sentiment
                return AdaptivePrompt(adaptive_prompt=base_prompt)
                
        except Exception as e:
            st.sidebar.error(f"Error generating adaptive prompt: {str(e)}")
            return AdaptivePrompt(adaptive_prompt=base_prompt)
    
    return RunnableLambda(generate_adaptive_prompt)

# Function to enforce hint-only approach in model responses (Adaptive Mode)
def enforce_hint_only_approach(response):
    # Check if the response is already a TutorResponse object
    if isinstance(response, TutorResponse):
        # Check if the hint appears to be giving a complete solution
        hint_text = response.hint.lower()
        solution_indicators = [
            "the solution is",
            "the answer is",
            "solving this completely",
            "to solve this problem",
            "we get",
            "therefore, x =",
            "therefore x =",
            "thus, x =",
            "thus x =",
            "x equals",
            "x is equal to"
        ]
        
        # If the hint contains solution indicators, replace it with a more appropriate hint
        for indicator in solution_indicators:
            if indicator in hint_text:
                return TutorResponse(
                    hint="I need to provide a hint rather than a complete solution. Let me give you a small hint for the next step: What specific technique could you apply here? Try to think about what we're trying to accomplish in this particular step. Can you attempt this step and share your thinking?",
                    reasoning="I detected that my response might be giving away too much of the solution. I should focus on providing guidance without solutions."
                )
        
        # If no solution indicators found, return the original response
        return response
    else:
        # If it's a string (from older versions), convert it to a TutorResponse
        try:
            if isinstance(response, str):
                return TutorResponse(
                    hint=response,
                    reasoning="This response was processed from a string format. I'm focusing on providing guidance without solutions."
                )
            else:
                # For any other type, return a default TutorResponse
                return TutorResponse(
                    hint="Let me give you a hint for this step. What approach do you think would be most appropriate here?",
                    reasoning="I'm providing a generic hint since the response format was unexpected."
                )
        except Exception as e:
            # If any error occurs, return a safe default
            return TutorResponse(
                hint="I need to provide a hint rather than a complete solution. What specific technique could you apply here?",
                reasoning="There was an error processing the response, but I should focus on providing guidance without solutions."
            )

# Main Streamlit UI
def main():
    st.title("Math Tutor")
    
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode / মোড নির্বাচন করুন",
        ["Interactive (Adaptive) / ইন্টারেক্টিভ (অভিযোজিত)", "Step-by-Step / ধাপে ধাপে"]
    )
    
    # Language selection
    language = st.sidebar.selectbox(
        "Select Language / ভাষা নির্বাচন করুন", 
        ["English", "Bangla"]
    )
    
    # Display appropriate header based on mode and language
    if mode == "Interactive (Adaptive) / ইন্টারেক্টিভ (অভিযোজিত)":
        if language == "Bangla":
            st.header("ইন্টারেক্টিভ গণিত শিক্ষক")
            st.markdown("এই টুলটি আপনাকে ইঙ্গিত এবং নির্দেশমূলক প্রশ্নের মাধ্যমে গণিত সমস্যা সমাধান করতে সাহায্য করবে। আপনার প্রতিক্রিয়ার উপর ভিত্তি করে, শিক্ষক আপনার বোঝার স্তর অনুযায়ী মানিয়ে নেবে।")
        else:
            st.header("Interactive Math Tutor")
            st.markdown("This tool will help you solve math problems through hints and guiding questions. Based on your responses, the tutor will adapt to your level of understanding.")
    else:
        if language == "Bangla":
            st.header("ধাপে ধাপে গণিত সমস্যা সমাধানকারী")
            st.markdown("এই টুলটি আপনাকে ধাপে ধাপে গণিত সমস্যা সমাধান করতে সাহায্য করবে। প্রতিটি ধাপের পরে, আপনি বুঝতে পেরেছেন কিনা তা জানাতে হবে।")
        else:
            st.header("Step-by-Step Math Problem Solver")
            st.markdown("This tool will help you solve math problems step by step. After each step, you'll need to indicate whether you understand.")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "adaptive_prompt" not in st.session_state:
        st.session_state.adaptive_prompt = ""
    
    if "sentiment" not in st.session_state:
        st.session_state.sentiment = "initial"
    
    if "last_tutor_hint" not in st.session_state:
        st.session_state.last_tutor_hint = ""
    
    if "consecutive_incorrect" not in st.session_state:
        st.session_state.consecutive_incorrect = 0
    
    if "current_step" not in st.session_state:
        st.session_state.current_step = 0
    
    if "problem_solved" not in st.session_state:
        st.session_state.problem_solved = False
    
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = mode
    
    if "current_language" not in st.session_state:
        st.session_state.current_language = language
    
    # Check if mode or language has changed
    if st.session_state.current_mode != mode or st.session_state.current_language != language:
        st.session_state.messages = []
        st.session_state.adaptive_prompt = ""
        st.session_state.sentiment = "initial"
        st.session_state.last_tutor_hint = ""
        st.session_state.consecutive_incorrect = 0
        st.session_state.current_step = 0
        st.session_state.problem_solved = False
        st.session_state.current_mode = mode
        st.session_state.current_language = language
    
    # Setup sidebar for Adaptive Mode
    if mode == "Interactive (Adaptive) / ইন্টারেক্টিভ (অভিযোজিত)":
        st.sidebar.title("Analysis Information")
        st.sidebar.subheader("Student Response Analysis")
        sentiment_display = st.sidebar.empty()
        
        st.sidebar.subheader("Adaptive Prompt")
        adaptive_prompt_display = st.sidebar.empty()
        
        st.sidebar.subheader("Hint Level")
        hint_level_display = st.sidebar.empty()
    
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if mode == "Interactive (Adaptive) / ইন্টারেক্টিভ (অভিযোজিত)":
                # For adaptive mode
                if message["role"] == "assistant" and "hint" in message and "reasoning" in message:
                    st.markdown(message["hint"])
                    with st.expander("See tutor's reasoning / শিক্ষকের যুক্তি দেখুন"):
                        st.markdown(message["reasoning"])
                else:
                    st.markdown(message["content"])
            else:
                # For step-by-step mode
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
            # Handle based on selected mode
            if mode == "Interactive (Adaptive) / ইন্টারেক্টিভ (অভিযোজিত)":
                # Adaptive Mode
                # Initialize LLM components
                chain, model, base_prompt = setup_adaptive_llm(language)
                
                # Create sentiment analysis chain
                sentiment_chain = create_sentiment_analysis_chain(model, language)
                sentiment_analyzer = create_sentiment_analyzer(sentiment_chain)
                adaptive_prompt_generator = create_adaptive_prompt_generator(model, base_prompt)
                runnable = setup_runnable(chain)
                
                # Determine if this is the first message or a response to a hint
                is_first_message = len([m for m in st.session_state.messages if m["role"] == "assistant"]) == 0
                
                if is_first_message:
                    # First message is a math problem, no sentiment analysis needed
                    current_system_prompt = base_prompt
                    st.session_state.sentiment = "initial"
                    if language == "Bangla":
                        sentiment_display.info("প্রাথমিক সমস্যা জমা")
                        adaptive_prompt_display.info("মূল শিক্ষক প্রম্পট ব্যবহার করছে")
                        hint_level_display.info("স্ট্যান্ডার্ড ইঙ্গিত স্তর")
                    else:
                        sentiment_display.info("Initial problem submission")
                        adaptive_prompt_display.info("Using base tutor prompt")
                        hint_level_display.info("Standard hint level")
                    st.session_state.consecutive_incorrect = 0
                else:
                    # Analyze the student's response to the previous hint
                    analysis_input = {
                        "student_response": prompt,
                        "tutor_hint": st.session_state.last_tutor_hint
                    }
                    
                    # Run sentiment analysis
                    analysis_result = sentiment_analyzer.invoke(analysis_input)
                    st.session_state.sentiment = analysis_result.get("sentiment", "unknown")
                    
                    # Update consecutive incorrect counter
                    if st.session_state.sentiment == "incorrect":
                        st.session_state.consecutive_incorrect += 1
                    else:
                        st.session_state.consecutive_incorrect = 0
                    
                    # Display sentiment in sidebar
                    sentiment_emoji = {
                        "correct": "✅",
                        "partially_correct": "⚠️",
                        "incorrect": "❌",
                        "unknown": "❓",
                        "initial": "🔍"
                    }
                    
                    if language == "Bangla":
                        sentiment_labels = {
                            "correct": "সঠিক",
                            "partially_correct": "আংশিকভাবে সঠিক",
                            "incorrect": "ভুল",
                            "unknown": "অজানা",
                            "initial": "প্রাথমিক"
                        }
                        sentiment_display.info(f"{sentiment_emoji.get(st.session_state.sentiment, '❓')} প্রতিক্রিয়া শ্রেণীবদ্ধ করা হয়েছে: {sentiment_labels.get(st.session_state.sentiment, 'অজানা')}")
                    else:
                        sentiment_display.info(f"{sentiment_emoji.get(st.session_state.sentiment, '❓')} Response classified as: {st.session_state.sentiment.replace('_', ' ').title()}")
                    
                    # Generate adaptive prompt based on sentiment
                    adaptive_result = adaptive_prompt_generator.invoke(analysis_result)
                    
                    # Enhance adaptive prompt based on consecutive incorrect responses
                    if st.session_state.consecutive_incorrect >= 2:
                        if language == "Bangla":
                            hint_level = f"উন্নত ইঙ্গিত স্তর ({st.session_state.consecutive_incorrect} ক্রমাগত ভুল প্রতিক্রিয়া)"
                            hint_level_display.warning(hint_level)
                            
                            # Add additional guidance for multiple incorrect responses
                            additional_guidance = f"""
                            গুরুত্বপূর্ণ: শিক্ষার্থী {st.session_state.consecutive_incorrect}টি ক্রমাগত ভুল প্রতিক্রিয়া দিয়েছে।
                            
                            আপনার করণীয়:
                            1. আরও স্পষ্ট ইঙ্গিত দিন, তবে এখনও সম্পূর্ণ সমাধান নয়
                            2. বর্তমান ধাপটিকে আরও ছোট, আরও সহজে পরিচালনাযোগ্য উপ-ধাপে ভাগ করুন
                            3. বিবেচনা করুন যদি শিক্ষার্থীর কোনো পূর্বশর্ত ধারণা অনুপস্থিত থাকে
                            4. মৌলিক বোঝাপড়া পরীক্ষা করার জন্য একটি খুব সাধারণ প্রশ্ন জিজ্ঞাসা করুন
                            5. একটি ভিন্ন পদ্ধতি বা ব্যাখ্যা শৈলী ব্যবহার করুন
                            
                            মনে রাখবেন: এমনকি সংগ্রামরত শিক্ষার্থীদের সাথেও, কখনই সম্পূর্ণ সমাধান প্রদান করবেন না।
                            নিশ্চিত করুন যে আপনার প্রতিক্রিয়া "hint" এবং "reasoning" ক্ষেত্র সহ একটি JSON হিসাবে কাঠামোবদ্ধ করা হয়েছে।
                            """
                        else:
                            hint_level = f"Enhanced hint level ({st.session_state.consecutive_incorrect} consecutive incorrect responses)"
                            hint_level_display.warning(hint_level)
                            
                            # Add additional guidance for multiple incorrect responses
                            additional_guidance = f"""
                            IMPORTANT: The student has given {st.session_state.consecutive_incorrect} consecutive incorrect responses.
                            
                            You should:
                            1. Provide a more explicit hint, but still not a complete solution
                            2. Break down the current step into smaller, more manageable sub-steps
                            3. Consider if there's a prerequisite concept the student is missing
                            4. Ask a very basic question to check fundamental understanding
                            5. Use a different approach or explanation style
                            
                            Remember: Even with struggling students, never provide the complete solution.
                            Make sure to structure your response as a JSON with "hint" and "reasoning" fields.
                            """
                        
                        adaptive_result.adaptive_prompt += additional_guidance
                    else:
                        if language == "Bangla":
                            hint_level_display.info("স্ট্যান্ডার্ড ইঙ্গিত স্তর")
                        else:
                            hint_level_display.info("Standard hint level")
                    
                    st.session_state.adaptive_prompt = adaptive_result.adaptive_prompt
                    
                    # Display adaptive prompt in sidebar
                    if language == "Bangla":
                        adaptive_prompt_display.info(f"উৎপন্ন অভিযোজিত প্রম্পট:\n\n{st.session_state.adaptive_prompt[:300]}...")
                    else:
                        adaptive_prompt_display.info(f"Generated adaptive prompt:\n\n{st.session_state.adaptive_prompt[:300]}...")
                    
                    # Combine base prompt with adaptive prompt
                    current_system_prompt = f"{base_prompt}\n\n{st.session_state.adaptive_prompt}"
                
                # Create a dynamic prompt template with the current system prompt
                dynamic_prompt = ChatPromptTemplate.from_messages([
                    ("system", current_system_prompt),
                    MessagesPlaceholder(variable_name="messages")
                ])
                
                # Create a new chain with the updated prompt
                dynamic_chain = dynamic_prompt | model
                dynamic_runnable = setup_runnable(dynamic_chain)
                
                # Invoke the chain with the updated prompt
                ai_message = dynamic_runnable.invoke(
                    [HumanMessage(content=prompt)],
                    config={"configurable": {"session_id": 'math_session'}}
                )
                
                # Process the response to extract hint and reasoning
                tutor_response = process_tutor_response_output(ai_message.content)
                
                # Enforce hint-only approach by checking and modifying the response if needed
                tutor_response = enforce_hint_only_approach(tutor_response)
                
                # Save the tutor's hint for next round of analysis
                st.session_state.last_tutor_hint = tutor_response.hint
                
                # Add AI message to session state with structured format
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": tutor_response.hint,  # For backward compatibility
                    "hint": tutor_response.hint,
                    "reasoning": tutor_response.reasoning
                })
                
                # Display the message with expandable reasoning
                with st.chat_message("assistant"):
                    # Display the hint with streaming effect
                    response = tutor_response.hint
                    
                    def stream_data():
                        for word in response.split(" "):
                            yield word + " "
                            time.sleep(0.015)
                    
                    st.write_stream(stream_data)
                    
                    # Display the reasoning in an expander
                    if language == "Bangla":
                        with st.expander("শিক্ষকের যুক্তি দেখুন"):
                            st.markdown(tutor_response.reasoning)
                    else:
                        with st.expander("See tutor's reasoning"):
                            st.markdown(tutor_response.reasoning)
            
            else:
                # Step-by-Step Mode
                # Initialize LLM components
                chain, model = setup_step_by_step_llm(language)
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
    if language == "Bangla":
        reset_text = "রিসেট করুন"
    else:
        reset_text = "Reset"
        
    if st.sidebar.button(reset_text):
        st.session_state.messages = []
        st.session_state.adaptive_prompt = ""
        st.session_state.sentiment = "initial"
        st.session_state.last_tutor_hint = ""
        st.session_state.consecutive_incorrect = 0
        st.session_state.current_step = 0
        st.session_state.problem_solved = False
        st.rerun()

if __name__ == "__main__":
    main()
