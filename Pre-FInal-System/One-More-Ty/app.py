import streamlit as st
import os
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
groq_api_key = os.getenv("GROQ_API_KEY")

# Define Pydantic models for structured output parsing
class StuOutputCorrectness(BaseModel):
    sentiment: Literal['correct', 'partially_correct', 'incorrect'] = Field(
        description='The sentiment classification of the student response'
    )

class AdaptivePrompt(BaseModel):
    adaptive_prompt: str = Field(
        description="The final adaptive prompt to guide the main model."
    )

# Initialize parsers
sentiment_parser = PydanticOutputParser(pydantic_object=StuOutputCorrectness)
adaptive_parser = PydanticOutputParser(pydantic_object=AdaptivePrompt)

# Initialize Langchain components
@st.cache_resource
def setup_llm():
    # Initialize the base model
    model = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.1)
    
    # Base prompt template (will be enhanced with adaptive prompts)
    base_prompt = """
    You are an AI math tutor specializing in high school algebra. Your goal is to guide students towards understanding and solving math problems independently by providing helpful hints and targeted questions.
    
    Your Primary Role: Providing Hints and Guiding Questions
    Receive the Problem: The student will give you an equation or math problem.
    Analyze the Problem: Briefly identify the core concepts involved (e.g., factoring quadratics, solving linear equations, applying the Pythagorean theorem).
    Provide a Hint (Instead of a Solution Step): Offer a single, targeted hint to nudge the student in the right direction. The hint should:
    Focus on a specific concept or technique relevant to the problem.
    Avoid giving away the complete solution to that step.
    Be phrased as a question or suggestion.
    Ask a Guiding Question: After providing the hint, always ask the student a question to encourage them to apply the hint. This question should:
    Directly relate to the hint you provided.
    Help the student think through the next step in the solution.
    Wait for Student Response: Do not solve the problem for the student. Wait for them to respond to your hint and question.
    Make sure that if student correct answer than you have to classify is correct.
    
    Important Principles:
    One Hint at a Time: Provide only one hint and one question per response. Avoid overwhelming the student with too much information.
    Focus on Understanding: Your goal is to help the student understand the underlying math concepts, not just memorize steps.
    Avoid Direct Answers: Never give the student the complete solution to a step unless they are completely stuck and have clearly demonstrated an inability to proceed after multiple hints.
    Be Patient: Allow the student time to think and respond.
    Do not ask general question like "do you understand?" or "does this make sense?". Always be specific. Like: "do you remember the square root of 4", etc.
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
             
             Provide the classification following this format:
             {format_instruction}"""),
        ]
    ).partial(format_instruction=sentiment_parser.get_format_instructions())
    
    # Create the sentiment analysis chain
    sentiment_chain = sentiment_prompt | model | sentiment_parser
    return sentiment_chain

# Setup adaptive prompt generators
def setup_adaptive_prompt_generators(model):
    # Prompt for correct responses
    prompt_correct = ChatPromptTemplate.from_messages(
        [
            ("system",
             """
             Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
             Think of yourself as a supervisor providing the best instruction for the AI tutor.
             
             The student gave a CORRECT response: {student_response}
             Previous tutor hint: {tutor_hint}
             
             Generate an adaptive prompt that:
             1. Acknowledges the student's correct understanding
             2. Encourages the student to move to the next step in the problem
             3. Maintains the tutoring approach of providing hints rather than solutions
             
             Return your result in the following format:
             {format_instructions}
             """)
        ]
    ).partial(format_instructions=adaptive_parser.get_format_instructions())
    
    # Prompt for partially correct responses
    prompt_partially_correct = ChatPromptTemplate.from_messages(
        [
            ("system",
             """
             Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
             Think of yourself as a supervisor providing the best instruction for the AI tutor.
             
             The student gave a PARTIALLY CORRECT response: {student_response}
             Previous tutor hint: {tutor_hint}
             
             Generate an adaptive prompt that:
             1. Acknowledges what the student understood correctly
             2. Identifies the specific misconception or error in their understanding
             3. Guides the tutor to provide a more targeted hint addressing this specific issue
             4. Maintains the tutoring approach of providing hints rather than solutions
             
             Return your result in the following format:
             {format_instructions}
             """)
        ]
    ).partial(format_instructions=adaptive_parser.get_format_instructions())
    
    # Prompt for incorrect responses
    prompt_incorrect = ChatPromptTemplate.from_messages(
        [
            ("system",
             """
             Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model.
             Think of yourself as a supervisor providing the best instruction for the AI tutor.
             
             The student gave an INCORRECT response: {student_response}
             Previous tutor hint: {tutor_hint}
             
             Generate an adaptive prompt that:
             1. Identifies the fundamental misunderstanding in the student's approach
             2. Guides the tutor to provide a more basic hint that addresses this misunderstanding
             3. Suggests a simpler question to check the student's understanding of prerequisites
             4. Maintains the tutoring approach of providing hints rather than solutions
             
             Return your result in the following format:
             {format_instructions}
             """)
        ]
    ).partial(format_instructions=adaptive_parser.get_format_instructions())
    
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

# Function to enrich input with sentiment analysis
def create_sentiment_analyzer(sentiment_chain):
    def analyze_sentiment(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Extract student response and tutor hint
        student_response = input_dict.get("student_response", "")
        tutor_hint = input_dict.get("tutor_hint", "")
        
        # Only analyze if there's a student response
        if student_response:
            try:
                sentiment_result = sentiment_chain.invoke({
                    "student_response": student_response,
                    "tutor_hint": tutor_hint
                })
                
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
    
    # Create the branch chain
    branch_chain = RunnableBranch(
        (lambda x: x.get("sentiment") == "correct", 
         prompt_correct | model | adaptive_parser),
        
        (lambda x: x.get("sentiment") == "partially_correct", 
         prompt_partially_correct | model | adaptive_parser),
        
        (lambda x: x.get("sentiment") == "incorrect", 
         prompt_incorrect | model | adaptive_parser),
        
        # Default branch if sentiment is not recognized
        RunnableLambda(lambda x: AdaptivePrompt(adaptive_prompt=base_prompt))
    )
    
    return branch_chain

# Main Streamlit UI
def main():
    st.title("Adaptive AI Math Tutor")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "adaptive_prompt" not in st.session_state:
        st.session_state.adaptive_prompt = ""
    
    if "sentiment" not in st.session_state:
        st.session_state.sentiment = "initial"
    
    if "last_tutor_hint" not in st.session_state:
        st.session_state.last_tutor_hint = ""
    
    # Setup sidebar
    st.sidebar.title("Analysis Information")
    st.sidebar.subheader("Student Response Analysis")
    sentiment_display = st.sidebar.empty()
    
    st.sidebar.subheader("Adaptive Prompt")
    adaptive_prompt_display = st.sidebar.empty()
    
    # Initialize LLM components
    chain, model, base_prompt = setup_llm()
    sentiment_chain = setup_sentiment_analyzer(model)
    sentiment_analyzer = create_sentiment_analyzer(sentiment_chain)
    adaptive_prompt_generator = create_adaptive_prompt_generator(model, base_prompt)
    runnable = setup_runnable(chain)
    
    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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
            else:
                # Analyze the student's response to the previous hint
                analysis_input = {
                    "student_response": prompt,
                    "tutor_hint": st.session_state.last_tutor_hint
                }
                
                # Run sentiment analysis
                analysis_result = sentiment_analyzer.invoke(analysis_input)
                st.session_state.sentiment = analysis_result.get("sentiment", "unknown")
                
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
                st.session_state.adaptive_prompt = adaptive_result.adaptive_prompt
                
                # Display adaptive prompt in sidebar
                adaptive_prompt_display.info(f"Generated adaptive prompt:\n\n{st.session_state.adaptive_prompt[:300]}...")
                
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
            ai_message = dynamic_runnable.invoke(
                [HumanMessage(content=prompt)],
                config={"configurable": {"session_id": 'math_session'}}
            )
            
            # Save the tutor's hint for next round of analysis
            st.session_state.last_tutor_hint = ai_message.content
            
            # Add AI message to session state
            st.session_state.messages.append({"role": "assistant", "content": ai_message.content})
            with st.chat_message("assistant"):
                st.markdown(ai_message.content)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your Groq API key and internet connection.")
            st.stop()

if __name__ == "__main__":
    main()
