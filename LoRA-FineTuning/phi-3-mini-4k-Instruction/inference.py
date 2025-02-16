import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Model and Tokenizer Loading (Load from Hugging Face Hub!)
model_name = "microsoft/Phi-3-mini-4k-instruct"  # This the base model
model_path = "alam1n/phi3-mini-algebra-tutor-v4"  # my finetuning model load from HF

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True  # or use quantization_config for newer versions
)
model = PeftModel.from_pretrained(model, model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get the special token IDs
end_token_id = tokenizer.convert_tokens_to_ids(["<|end|>"])[0]
user_token_id = tokenizer.convert_tokens_to_ids(["<|user|>"])[0]
assistant_token_id = tokenizer.convert_tokens_to_ids(["<|assistant|>"])[0]


# 2. Modified Inference Function for Interactive Dialogue
def generate_next_step(prompt):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            max_new_tokens=200,  # Adjusted to 100, experiment with this
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.15,
            eos_token_id=end_token_id,  # Use the correct <|end|> token ID
            pad_token_id=end_token_id # Use the correct <|end|> token ID
        )

    generated_text = tokenizer.decode(outputs[:, input_ids["input_ids"].shape[-1]:][0], skip_special_tokens=False) #Keep special tokens
    # Splitting and cleaning the response
    generated_text = generated_text.split("<|end|>")[0].strip() #take only assistant
    return generated_text

# 3. Interactive Dialogue Loop
def interactive_session(initial_prompt):
    prompt = initial_prompt
    while True:
        response = generate_next_step(prompt)
        print(response)  # Print the model's response
        print("\n")

        user_input = input("Your response (or type 'exit' to quit): ")  # Get user input

        if user_input.lower() == "exit":
            break

        # Append the model's response and the user's input to the prompt. Add special token
        prompt += response + "<|end|>\n<|user|>\n" + user_input + "<|end|>\n<|assistant|>\n"
        #print("current prompt", prompt)


# Example Usage:
initial_prompt = "<|system|>\nYou are a helpful tutor for solving algebra.\n<|end|>\n\n<|user|>\nSimplify x^3 + 7x^2 - 2x^3.\n<|end|>\n\n<|assistant|>\n"
# initial_prompt = "<|system|>\nYou are a helpful tutor for solving algebra.\n<|end|>\n\n<|user|>\nSolve the following equations 4x-2(x+1) = 5(x + 3) + 5.\n<|end|>\n\n<|assistant|>\n"
interactive_session(initial_prompt)