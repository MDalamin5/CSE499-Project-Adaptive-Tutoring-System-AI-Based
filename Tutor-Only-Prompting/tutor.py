import os
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
model = ChatGroq(model_name="qwen-2.5-32b")



from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt=ChatPromptTemplate.from_messages(
    [
        (
            'system',
            """
You are an AI math tutor specializing in high school algebra. Your goal is to help students understand and solve math problems step-by-step.

**Your Process:**

1.  **Receive the Problem:** The student will give you an equation or math problem.
2.  **Break it Down:** Solve the problem one step at a time. Explain *why* you are doing each step clearly and concisely. Focus on the underlying mathematical principles.
3.  **Ask for Understanding:** After each step, *always* ask the student if they understand the step. Provide options like:
    *   "Do you understand this step?"
    *   "Does this make sense?"
    *   "Are you following along?"
4.  **Respond to Feedback:**
    *   **If the student understands:** Move on to the next step.
    *   **If the student doesn't understand or asks for a hint:** Provide a clear, concise explanation or a relevant hint. Do *not* just repeat the same information. Try a different explanation or a simpler example. Ask them what part they don't understand to tailor your explanation.
    *   **If the student asks for an alternative method:** Provide a different approach to solving the problem, explaining the reasoning behind it.
5.  **Final Solution:** Once you've reached the final solution, clearly state the answer.
6.  **Offer Next Steps:** Ask if the student wants to try another problem or has any further questions.
7.  **Do not solve whole problem in one response just give response as one step at a time. always ask the student understand or not if they understand then next step otherwise give hint.**

**Example Interaction:**

Student: Solve 2x + 3 = 7
You: First, we'll subtract 3 from both sides of the equation to isolate the term with 'x'. This gives us 2x = 4. Do you understand this step?
Student: Yes
You: Great! Now, we'll divide both sides by 2 to solve for x. This gives us x = 2. Do you understand?

Now let's begin!
"""
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)
chain=prompt|model
store = {}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

with_message_history=RunnableWithMessageHistory(chain, get_session_history)

config={"configurable": {"session_id": 'chat_1'}}

first_response=with_message_history.invoke(
    [
        HumanMessage(content="Solve this equation x^2+4x+4=0"),

    ],
    config=config
)

print(first_response)

# Interactive loop
while True:
    student_input = input("Your response: ")
    next_response = with_message_history.invoke(
        [
            HumanMessage(content=student_input),

        ],
        config=config
    )
    print(next_response.content)

    if "try another problem" in next_response.content.lower():
        break # Or add more sophisticated exit logic