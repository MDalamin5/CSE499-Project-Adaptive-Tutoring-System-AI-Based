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
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, List

# Load environment variables
load_dotenv()

# its for local machine
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Interactive Way Math Problem Solving"

# Access the Groq API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Set environment variables for Langchain tracing (if you're using it)
# os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "") # Get the value if it exists
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Interactive Way Math Problem Solving"
# groq_api_key = os.getenv("GROQ_API_KEY") # its for local machine

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
                    if self.model_class == AdaptivePrompt:
                        return AdaptivePrompt(adaptive_prompt="The model output could not be parsed correctly. Please provide a hint for the current step without giving away the solution.")
                    elif self.model_class == StuOutputCorrectness:
                        return StuOutputCorrectness(sentiment="partially_correct")
                    elif self.model_class == TutorResponse:
                        return TutorResponse(
                            hint="I need to provide a hint rather than a complete solution. What specific technique could you apply here?",
                            reasoning="The parsing failed, but I should focus on providing guidance without solutions."
                        )
        else:
            # If no JSON-like structure is found, create a default instance
            if self.model_class == AdaptivePrompt:
                return AdaptivePrompt(adaptive_prompt="The model output could not be parsed correctly. Please provide a hint for the current step without giving away the solution.")
            elif self.model_class == StuOutputCorrectness:
                return StuOutputCorrectness(sentiment="partially_correct")
            elif self.model_class == TutorResponse:
                return TutorResponse(
                    hint="I need to provide a hint rather than a complete solution. What specific technique could you apply here?",
                    reasoning="No structured output was found, but I should focus on providing guidance without solutions."
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
sentiment_parser = PydanticOutputParser(pydantic_object=StuOutputCorrectness)
adaptive_parser = PydanticOutputParser(pydantic_object=AdaptivePrompt)
tutor_response_parser = PydanticOutputParser(pydantic_object=TutorResponse)

# Create robust parsers
robust_sentiment_parser = RobustOutputParser(sentiment_parser)
robust_adaptive_parser = RobustOutputParser(adaptive_parser)
robust_tutor_response_parser = RobustOutputParser(tutor_response_parser)

# Initialize Langchain components
@st.cache_resource
def setup_llm():
    # Initialize the base model
    model = ChatGroq(model_name="qwen-qwq-32b", temperature=0.1, groq_api_key=groq_api_key)
    
    # Enhanced base prompt template with stronger emphasis on hints-only approach
    base_prompt = """
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
    
    # Create the prompt template with placeholder for adaptive prompts
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "{system_prompt}"),
            MessagesPlaceholder(variable_name="messages")
        ]
    ).partial(system_prompt=base_prompt)
    
    # Create the chain
    chain = prompt | model
    return chain, model, base_prompt

# Setup sentiment analysis chain
def setup_sentiment_analyzer(model):
    # Create prompt template for sentiment analysis
    sentiment_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
             """Analyze the following student response to a math problem and classify its correctness.
             Determine whether it is 'correct', 'partially_correct', or 'incorrect'.
             
             Previous tutor hint: {tutor_hint}
             Student Response: {student_response}
             
             IMPORTANT: Your response must be a valid JSON object with the following format:
             {{
                 "sentiment": "correct" OR "partially_correct" OR "incorrect"
             }}
             
             Do not include any explanations, thinking, or additional text outside the JSON object.
             """),
        ]
    )
    
    # Create the sentiment analysis chain with robust parsing
    sentiment_chain = sentiment_prompt | model 
    return sentiment_chain

# Setup adaptive prompt generators
def setup_adaptive_prompt_generators(model):
    # Common instructions for all prompt types
    json_format_instructions = """
    IMPORTANT: Your response must be a valid JSON object with the following format:
    {{
        "adaptive_prompt": "your adaptive prompt text here"
    }}
    
    Do not include any explanations, thinking, or additional text outside the JSON object.
    """
    
    # Prompt for correct responses
    prompt_correct = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"""
             Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
             Think of yourself as a supervisor providing the best instruction for the AI tutor.
             
             The student gave a CORRECT response: {{student_response}}
             Previous tutor hint: {{tutor_hint}}
             
             Generate an adaptive prompt that:
             1. Acknowledges the student's correct understanding
             2. Instructs the tutor to move to the NEXT STEP ONLY (not multiple steps)
             3. Emphasizes that the tutor should provide only a hint for this next step, not a solution
             4. Reminds the tutor to ask a guiding question about this next step
             5. Reinforces that the tutor must NEVER provide complete solutions
             6. Reminds the tutor to structure the response as a JSON with "hint" and "reasoning" fields
             
             {json_format_instructions}
             """)
        ]
    )
    
    # Prompt for partially correct responses
    prompt_partially_correct = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"""
             Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
             Think of yourself as a supervisor providing the best instruction for the AI tutor.
             
             The student gave a PARTIALLY CORRECT response: {{student_response}}
             Previous tutor hint: {{tutor_hint}}
             
             Generate an adaptive prompt that:
             1. Acknowledges what the student understood correctly
             2. Identifies the specific misconception or error in their understanding
             3. Instructs the tutor to provide a more targeted hint addressing this specific issue
             4. Emphasizes that the tutor should NOT move to the next step yet
             5. Reminds the tutor to ask a more specific guiding question
             6. Reinforces that the tutor must NEVER provide complete solutions
             7. Reminds the tutor to structure the response as a JSON with "hint" and "reasoning" fields
             
             {json_format_instructions}
             """)
        ]
    )
    
    # Prompt for incorrect responses
    prompt_incorrect = ChatPromptTemplate.from_messages(
        [
            ("system",
             f"""
             Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
             Think of yourself as a supervisor providing the best instruction for the AI tutor.
             
             The student gave an INCORRECT response: {{student_response}}
             Previous tutor hint: {{tutor_hint}}
             
             Generate an adaptive prompt that:
             1. Identifies the fundamental misunderstanding in the student's approach
             2. Instructs the tutor to provide a more basic hint that addresses this misunderstanding
             3. Directs the tutor to break down the current step into smaller sub-steps
             4. Suggests a simpler question to check the student's understanding of prerequisites
             5. Emphasizes that the tutor should NOT move to the next step yet
             6. Reinforces that the tutor must NEVER provide complete solutions, even if the student is struggling
             7. Reminds the tutor to structure the response as a JSON with "hint" and "reasoning" fields
             
             {json_format_instructions}
             """)
        ]
    )
    
    return prompt_correct, prompt_partially_correct, prompt_incorrect

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

# Function to process model output for sentiment analysis
def process_sentiment_output(output_text):
    try:
        return robust_sentiment_parser.parse(output_text)
    except Exception as e:
        st.sidebar.error(f"Error parsing sentiment: {str(e)}")
        return StuOutputCorrectness(sentiment="partially_correct")

# Function to process model output for adaptive prompt
def process_adaptive_output(output_text):
    try:
        return robust_adaptive_parser.parse(output_text)
    except Exception as e:
        st.sidebar.error(f"Error parsing adaptive prompt: {str(e)}")
        return AdaptivePrompt(adaptive_prompt="The model output could not be parsed correctly. Please provide a hint for the current step without giving away the solution.")

# Function to process model output for tutor response
def process_tutor_response_output(output_text):
    try:
        return robust_tutor_response_parser.parse(output_text)
    except Exception as e:
        st.sidebar.error(f"Error parsing tutor response: {str(e)}")
        return TutorResponse(
            hint="I need to provide a hint rather than a complete solution. What specific technique could you apply here?",
            reasoning="The parsing failed, but I should focus on providing guidance without solutions."
        )

# Function to enrich input with sentiment analysis
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

# Function to create the adaptive prompt generator
def create_adaptive_prompt_generator(model, base_prompt):
    # Setup the prompt generators
    prompt_correct, prompt_partially_correct, prompt_incorrect = setup_adaptive_prompt_generators(model)
    
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

# Function to enforce hint-only approach in model responses
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
    st.set_page_config(
    page_title="Adaptive Tutor",
    page_icon="💡",
    layout="wide"
    )
    st.title("Adaptive AI Math Tutor")
    st.write(
     """
     Welcome! I'm your personalized `AI Math Tutor`, designed to guide you
     through challenging problems with helpful hints and targeted questions.
     Let's tackle together 🤝!
     """
    )
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
    
    # Setup sidebar
    st.sidebar.title("Analysis Information")
    st.sidebar.subheader("Student Response Analysis")
    sentiment_display = st.sidebar.empty()
    
    st.sidebar.subheader("Adaptive Prompt")
    adaptive_prompt_display = st.sidebar.empty()
    
    st.sidebar.subheader("Hint Level")
    hint_level_display = st.sidebar.empty()
    
    # Initialize LLM components
    chain, model, base_prompt = setup_llm()
    sentiment_chain = setup_sentiment_analyzer(model)
    sentiment_analyzer = create_sentiment_analyzer(sentiment_chain)
    adaptive_prompt_generator = create_adaptive_prompt_generator(model, base_prompt)
    runnable = setup_runnable(chain)
    
    # Display existing messages
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                # For assistant messages, check if we have hint and reasoning
                if "hint" in message and "reasoning" in message:
                    st.markdown(message["hint"])
                    with st.expander("See tutor's reasoning"):
                        st.markdown(message["reasoning"])
                else:
                    # Fallback for messages without structured format
                    st.markdown(message["content"])
    
    # Get user input
    if prompt := st.chat_input("Enter your math problem or response here..."):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            # Determine if this is the first message or a response to a hint
            is_first_message = len(st.session_state.messages) <= 1
            
            if is_first_message:
                # First message is a math problem, no sentiment analysis needed
                current_system_prompt = base_prompt
                st.session_state.sentiment = "initial"
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
                
                sentiment_display.info(f"{sentiment_emoji.get(st.session_state.sentiment, '❓')} Response classified as: {st.session_state.sentiment.replace('_', ' ').title()}")
                
                # Generate adaptive prompt based on sentiment
                adaptive_result = adaptive_prompt_generator.invoke(analysis_result)
                
                # Enhance adaptive prompt based on consecutive incorrect responses
                if st.session_state.consecutive_incorrect >= 2:
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
                    hint_level_display.info("Standard hint level")
                
                st.session_state.adaptive_prompt = adaptive_result.adaptive_prompt
                
                # Display adaptive prompt in sidebar
                adaptive_prompt_display.info(f"Generated adaptive prompt:\n\n{st.session_state.adaptive_prompt[:400]}...")
                
                # Combine base prompt with adaptive prompt
                current_system_prompt = f"{base_prompt}\n\n{st.session_state.adaptive_prompt}"
            
            # Create a dynamic prompt template with the current system prompt
            dynamic_prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', current_system_prompt),
                    MessagesPlaceholder(variable_name="messages")
                ]
            )
            
            # Create a new chain with the updated prompt
            dynamic_chain = dynamic_prompt | model
            dynamic_runnable = setup_runnable(dynamic_chain)
            
            # Invoke the chain with the updated prompt
            with st.spinner("Tutor is thinking..."):
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
                st.markdown(tutor_response.hint)
                with st.expander("See tutor's reasoning"):
                    st.markdown(tutor_response.reasoning)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your Groq API key and internet connection.")
            st.stop()

if __name__ == "__main__":
    main()
