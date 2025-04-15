import streamlit as st
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

# ----- Initialize the model  -----
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")

# ----- Define Data Models and Parsers -----
class StuOutputCorrectness(BaseModel):
    sentiment: Literal['correct', 'partially_correct', 'incorrect'] = Field(
        description='Give the sentiment from the student response')

parser = PydanticOutputParser(pydantic_object=StuOutputCorrectness)

class AdaptivePrompt(BaseModel):
    adaptive_prompt: str = Field(description="The final adaptive prompt to guide the main model.")

adaptive_parser = PydanticOutputParser(pydantic_object=AdaptivePrompt)

# ----- Define Prompts and Chains -----
system_messages = "You are a helpful assistant."

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_messages),
        ("human",
         "Analyze the following student response and classify its correctness. "
         "Determine whether it is 'correct', 'partially_correct', or 'incorrect'.\n\n"
         "Student Response: {response}\n\n"
         "Provide the classification following this format:\n{format_instruction}")
    ]
).partial(format_instruction=parser.get_format_instructions())

classifier_chain = chat_prompt | model | parser

prompt_correct = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Student gave the correct response and the response is {response}. "
         "Now give an adaptive prompt so the model can generate only the best response for the next step. "
         "Also analyze previous message history — it will help generate a better adaptive prompt."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

prompt_partially_correct = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         your task is generate a best adaptive prompt. And this adaptive prompt goes to my main model as the system prompt. Using this prompt the main model will be generated the elaborated response you just generate one the adaptive prompt only. Think you just help to main model as a instructor like a boss.
         for generate one best adaptive prompt you can analysis the previous chat and current response also.
         Student gave a response which is partially correct and the response is {response}. 
         """
         ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

prompt_incorrect = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Student gave a totally incorrect response and the response is {response}. "
         "Analyze the student’s misunderstanding — maybe they need clarification on core concepts. "
         "Provide elaborated hints or ask guiding questions. "
         "Use previous message history to craft the best adaptive prompt for the next step."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

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
         """
         ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(format_instructions=adaptive_parser.get_format_instructions())


branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == 'correct', prompt_correct | model | adaptive_parser),
    (lambda x: x["sentiment"] == 'partially_correct', prompt_partially_correct | model | adaptive_parser),
    (lambda x: x["sentiment"] == 'incorrect', prompt_partially_correct | model | adaptive_parser),
    RunnableLambda(lambda x: {"adaptive_prompt": "Couldn't determine sentiment."})
)

def enrich_with_sentiment(input: dict) -> dict:
    sentiment_result = classifier_chain.invoke(input)
    print(sentiment_result)
    return {
        **input,  # includes 'messages' and 'response'
        'sentiment': sentiment_result.sentiment
    }

enriched_chain = RunnableLambda(enrich_with_sentiment)

final_chain = enriched_chain | branch_chain


# ----- Streamlit App -----

st.title("AI-Powered Adaptive Tutor")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize adaptive prompt in session state
if "adaptive_prompt" not in st.session_state:
    st.session_state.adaptive_prompt = system_messages  # Start with the base system message

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Student input
if prompt := st.chat_input("Enter your question/problem:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- First Turn: Give initial hint
    if len(st.session_state.messages) == 1: # It's the first turn
        with st.chat_message("assistant"):
            hint = model.invoke( [HumanMessage(content=prompt)])
            st.markdown(hint.content)
        st.session_state.messages.append({"role": "assistant", "content": hint.content})

    # --- Subsequent Turns: Adaptive Prompting
    else:
        # 1. Get student's latest response
        student_response = st.session_state.messages[-1]["content"]

        # 2. Prepare input for the chain
        chain_input = {
            "messages": st.session_state.messages[:-1],  # All previous messages
            "response": student_response
        }

        # 3. Invoke the adaptive prompt chain
        adaptive_result = final_chain.invoke(chain_input)
        print("Adaptive Result:", adaptive_result)

        # 4. Extract the adaptive prompt
        adaptive_prompt = adaptive_result.get("adaptive_prompt")

        # 5. Display the adaptive prompt in the sidebar
        with st.sidebar:
            st.subheader("Adaptive Prompt")
            st.write(adaptive_prompt)

        # 6. Generate response using the adaptive prompt
        with st.chat_message("assistant"):
            # Use session state to store messages to persist them across reruns
            system_message = SystemMessage(content=st.session_state.adaptive_prompt)
            human_message = HumanMessage(content=prompt)
            response = model.invoke([system_message, human_message])
            st.markdown(response.content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.content})

        # update adaptive prompt
        st.session_state.adaptive_prompt = adaptive_prompt