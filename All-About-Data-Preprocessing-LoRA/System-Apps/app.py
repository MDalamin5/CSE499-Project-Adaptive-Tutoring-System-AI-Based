import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Model Loading (Cached for Efficiency) ---
@st.cache_resource
def load_model():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    model_path = "alam1n/phi3-mini-algebra-tutor-v4"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get the special token IDs (do this only once)
    end_token_id = tokenizer.convert_tokens_to_ids(["<|end|>"])[0]
    user_token_id = tokenizer.convert_tokens_to_ids(["<|user|>"])[0]
    assistant_token_id = tokenizer.convert_tokens_to_ids(["<|assistant|>"])[0]

    return model, tokenizer, end_token_id

model, tokenizer, end_token_id = load_model()

# --- Inference Function ---
def generate_next_step(prompt, model, tokenizer, end_token_id):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.15,
            eos_token_id=end_token_id,
            pad_token_id=end_token_id
        )

    generated_text = tokenizer.decode(outputs[:, input_ids["input_ids"].shape[-1]:][0], skip_special_tokens=False)
    generated_text = generated_text.split("<|end|>")[0].strip()
    return generated_text


# --- Streamlit App ---
st.title("Algebra Tutor Chatbot")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = "<|system|>\nYour name is DeltaAlgebra. You are a friendly and patient math tutor. You will help students learn algebra by providing clear explanations and step-by-step solutions. You will offer hints when needed and identify errors in their work. You will encourage them to keep trying.\n<|end|>\n\n<|user|>\nHello!\n<|end|>\n\n<|assistant|>\n"
    st.session_state.messages = [{"role": "tutor", "content": "Hello! How can I help you today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user message
user_input = st.chat_input("Your Question/Greeting:")

# React to user input
if user_input:
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    prompt = st.session_state.conversation_history + user_input + "<|end|>\n<|assistant|>\n"
    response = generate_next_step(prompt, model, tokenizer, end_token_id)

    # Display assistant response in chat message container
    st.session_state.messages.append({"role": "tutor", "content": response})
    with st.chat_message("tutor"):
        st.markdown(response)

    # Update conversation history
    st.session_state.conversation_history = prompt + response + "<|end|>\n<|user|>\n"