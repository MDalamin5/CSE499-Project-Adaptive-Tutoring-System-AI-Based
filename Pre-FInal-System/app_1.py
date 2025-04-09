import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableBranch
from pydantic import BaseModel, Field
from typing import Literal
import json

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Define Pydantic models for adaptive prompt generation (assuming these are correct)
class StuOutputCorrectness(BaseModel):
    sentiment: Literal['correct', 'partially_correct', 'incorrect'] = Field(description='Give the sentiment from the student response')

class AdaptivePrompt(BaseModel):
    adaptive_prompt: str = Field(description="The final adaptive prompt to guide the main model.")

# --- Adaptive Prompt Generation Components (Assuming these are already defined and functional) ---
# (You should have your classifier_chain, prompt_correct, prompt_partially_correct, prompt_incorrect,
#  branch_chain, and enrich_with_sentiment defined as in your original code.)
#  I am only including the definitions of the functions you provided earlier to demonstrate the
#  adaptive prompt integration. You will need to supply the others yourself.

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser


groq_api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(groq_api_key=groq_api_key,model_name="qwen-2.5-32b")

class StuOutputCorrectness(BaseModel):
    sentiment: Literal['correct', 'partially_correct', 'incorrect'] = Field(description='Give the sentiment from the student response *in the context of the tutor question*')
    

class AdaptivePrompt(BaseModel):
    adaptive_prompt: str = Field(description="The final adaptive prompt to guide the main model.")
    
parser = PydanticOutputParser(pydantic_object=StuOutputCorrectness)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Analyze the following interaction between a tutor and a student and classify the student's response.
        Determine whether the student response is 'correct', 'partially_correct', or 'incorrect' *in the context of the tutor's question related to math*.

        Tutor's Question: {tutor_question}
        Student Response: {response}

        Provide the classification following this format:\n{format_instruction}"""),
        MessagesPlaceholder(variable_name='messages')
    ]
).partial(format_instruction=parser.get_format_instructions())

classifier_chain = chat_prompt | model | parser


## branching prompt

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
adaptive_parser = PydanticOutputParser(pydantic_object=AdaptivePrompt)


prompt_correct = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model. 
         Think of yourself as a supervisor providing the best instruction for the AI tutor.
        
         Student gave a response which is correct: {response}
         You can analyze the previous chat history and the student’s latest response.
         Return your result in the following format:\n{format_instructions}

         Make sure that the adaptive_prompt value is in correct json format like \\\\, \\n etc.
         """
         
         ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(format_instructions=adaptive_parser.get_format_instructions())



prompt_partially_correct = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model. 
         Think of yourself as a supervisor providing the best instruction for the AI tutor.

         You can analyze the previous chat history and the student’s latest response.

         Student gave a response which is partially correct: {response}

         Return your result in the following format:
         {format_instructions}
         Make sure that the adaptive_prompt value is in correct json format like \\\\, \\n etc.
         """
         ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(format_instructions=adaptive_parser.get_format_instructions())

prompt_incorrect = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """
         Your task is to generate one adaptive prompt only. This prompt will be used as the system prompt for the main model. 
         Think of yourself as a supervisor providing the best instruction for the AI tutor.
        
         Student gave a response which is incorrect: {response}
         You can analyze the previous chat history and the student’s latest response.
         Return your result in the following format:\n{format_instructions}
         Make sure that the adaptive_prompt value is in correct json format like \\\\, \\n etc.
         """
         ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(format_instructions=adaptive_parser.get_format_instructions())

## hare i just update the partially correct prompt which can produce the pydantic output you update correct and incorrect prompt also

def create_adaptive_prompt(chain):
    def get_prompt(input):
        result = chain.invoke(input)
        return {
            "adaptive_prompt": result.adaptive_prompt,
            "sentiment": input["sentiment"]
        }
    return get_prompt


branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == 'correct', create_adaptive_prompt(prompt_correct | model | adaptive_parser)),
    (lambda x: x["sentiment"] == 'partially_correct', create_adaptive_prompt(prompt_partially_correct | model | adaptive_parser)),
    (lambda x: x["sentiment"] == 'incorrect', create_adaptive_prompt(prompt_incorrect | model | adaptive_parser)),
    RunnableLambda(lambda x: {"adaptive_prompt": "Couldn't determine sentiment.", "sentiment": "unknown"})
)

from langchain_core.runnables import RunnableLambda

# Wrap classifier_chain to return full input + parsed sentiment
def enrich_with_sentiment(input: dict) -> dict:
    messages = input['messages']
    student_response = input['response']
    if len(messages) >= 2:
        tutor_question = messages[-2]['content'] # Access Tutor's question here
    else:
        tutor_question = "No previous question available."

    sentiment_result = classifier_chain.invoke({
        'response': student_response,
        'tutor_question': tutor_question, # and here
        'messages': messages  # Still pass the full history for context in other chains
    })
    print(sentiment_result)
    return {
        **input,
        'sentiment': sentiment_result.sentiment
    }

enriched_chain = RunnableLambda(enrich_with_sentiment)

final_chain = enriched_chain | branch_chain
# --- End Adaptive Prompt Generation Components ---

# Initialize Langchain components
@st.cache_resource
def setup_llm(adaptive_prompt=""): # Added adaptive_prompt as argument with default value
    model = ChatGroq(model_name="qwen-2.5-32b", temperature=0.1) # Lower temperature for more predictable responses

    # Hint-focused system prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                f"""
You are an AI math tutor specializing in high school algebra. Your goal is to guide students towards understanding and solving math problems independently by providing helpful hints and targeted questions.

**Your Primary Role: Providing Hints and Guiding Questions**

1.  **Receive the Problem:** The student will give you an equation or math problem.
2.  **Analyze the Problem:** Briefly identify the core concepts involved (e.g., factoring quadratics, solving linear equations, applying the Pythagorean theorem).
3.  **Provide a Hint (Instead of a Solution Step):** Offer a *single, targeted hint* to nudge the student in the right direction. The hint should:
    *   Focus on a specific concept or technique relevant to the problem.
    *   Avoid giving away the complete solution to that step.
    *   Be phrased as a question or suggestion.
4.  **Ask a Guiding Question:** After providing the hint, *always* ask the student a question to encourage them to apply the hint. This question should:
    *   Directly relate to the hint you provided.
    *   Help the student think through the next step in the solution.
    5.  **Wait for Student Response:** Do *not* solve the problem for the student. Wait for them to respond to your hint and question.
    Make sure that if student correct answer than you have to classify is correct.
6.  **Example Interaction:**

    *   Student: Solve x^2 + 5x + 6 = 0
    *   You: This problem involves factoring a quadratic expression. Can you think of two numbers that multiply to 6 and add up to 5?
    *   Student: [Responds with their attempt or confusion]

**Important Principles:**

*   **One Hint at a Time:** Provide only *one* hint and one question per response. Avoid overwhelming the student with too much information.
*   **Focus on Understanding:** Your goal is to help the student *understand* the underlying math concepts, not just memorize steps.
*   **Avoid Direct Answers:** Never give the student the complete solution to a step unless they are completely stuck and have clearly demonstrated an inability to proceed after multiple hints.
*   **Be Patient:** Allow the student time to think and respond.
*   **Do not ask general question like "do you understand?" or "does this make sense?". Always be specific. Like: "do you remember the square root of 4", etc.**

{'**Adaptive Guidance:** ' + adaptive_prompt if adaptive_prompt else ''}

Now let's begin!
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
st.title("AI Math Tutor")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Enter your math problem here..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine if this is the first turn
    is_first_turn = len(st.session_state.messages) == 1 # Only the user's initial problem is in the history

    # Generate adaptive prompt only if it's NOT the first turn
    if not is_first_turn:
        # Generate adaptive prompt
        adaptive_prompt_result = final_chain.invoke({
            'messages': st.session_state.messages[:-1],  # All messages *except* the latest user input
            'response': prompt  # User's latest input (the response to be analyzed)
        })
        try:
            adaptive_prompt = adaptive_prompt_result["adaptive_prompt"]
            sentiment = adaptive_prompt_result["sentiment"]  #Extract sentiment

        except (AttributeError, ValueError, TypeError) as e:
            st.error(f"Error processing adaptive prompt: {e}")
            adaptive_prompt = ""
            sentiment = "Error"

    else:
        adaptive_prompt = "" # No adaptive prompt on the first turn
        sentiment = "N/A"


    # Display message history and sentiment in the sidebar
    st.sidebar.header("Message History:")
    for msg in st.session_state.messages:
        st.sidebar.write(f"**{msg['role']}:** {msg['content']}")
    st.sidebar.write(f"**Sentiment:** {sentiment}")
    
    st.sidebar.write(f"Adaptive Prompt: {adaptive_prompt}")

    # Initialize LLM chain with the adaptive prompt (even if it's empty on the first turn)
    chain = setup_llm(adaptive_prompt)
    runnable = setup_runnable(chain)

    # Invoke the chain and get the AI response
    try:
        ai_message = runnable.invoke(
            [HumanMessage(content=prompt)], # Pass only the latest user input
            config={"configurable": {"session_id": 'math_session'}}
        )
        # Add AI message to session state
        st.session_state.messages.append({"role": "assistant", "content": ai_message.content})
        with st.chat_message("assistant"):
            st.markdown(ai_message.content)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please check your Groq API key and internet connection.")
        st.stop()