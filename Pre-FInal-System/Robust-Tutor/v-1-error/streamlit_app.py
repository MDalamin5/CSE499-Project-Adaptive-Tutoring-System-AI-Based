import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Get the API key from environment variables or use a default placeholder
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Create a .env file with the API key
def save_api_key(api_key):
    with open(".env", "w") as f:
        f.write(f"GROQ_API_KEY={api_key}")
    os.environ["GROQ_API_KEY"] = api_key
    st.success("API key saved successfully!")
    st.rerun()

# Set page config
st.set_page_config(
    page_title="Dual-Mode Math Tutor",
    page_icon="🧮",
    layout="wide"
)

# Main app
st.title("Dual-Mode Math Tutor - Configuration")

# API key input
st.header("API Key Configuration")
st.write("To use this application, you need to provide a Groq API key.")

# Check if API key is already set
if GROQ_API_KEY:
    st.success("API key is configured!")
    if st.button("Change API Key"):
        # If user wants to change the key, show the input field
        new_api_key = st.text_input("Enter your Groq API key:", type="password")
        if new_api_key and st.button("Save New API Key"):
            save_api_key(new_api_key)
else:
    # If no API key is set, show the input field
    api_key = st.text_input("Enter your Groq API key:", type="password")
    if api_key and st.button("Save API Key"):
        save_api_key(api_key)

# Instructions
st.header("Instructions")
st.write("""
1. Get a Groq API key from [https://console.groq.com/](https://console.groq.com/)
2. Enter your API key in the field above and click 'Save API Key'
3. Once configured, you'll be able to use the Dual-Mode Math Tutor
""")

# Welcome message and features
st.header("Welcome to the Dual-Mode Math Tutor")

st.markdown("""
## Two Different Ways to Learn Mathematics

The Dual-Mode Math Tutor offers two distinct approaches to help you learn mathematics:

### 1. Interactive (Adaptive) Mode

This mode provides a highly personalized learning experience:
- Analyzes your responses to determine your understanding level
- Adapts the guidance based on whether your answers are correct, partially correct, or incorrect
- Provides increasingly detailed hints if you're struggling with a concept
- Shows the tutor's reasoning in expandable sections
- Available in both English and Bangla

### 2. Step-by-Step Mode

This mode offers a structured, methodical approach:
- Breaks down problems into clear, sequential steps
- Checks your understanding after each step before proceeding
- Provides visual indicators when the solution is complete
- Available in both English and Bangla

### How to Use:

1. Configure your API key above (if you haven't already)
2. Launch the Math Tutor using the button below
3. Select your preferred mode and language in the sidebar
4. Enter a math problem to begin
5. Follow the tutor's guidance to solve the problem

### Example Problems to Try:

- Solve quadratic equations (e.g., x² + 5x + 6 = 0)
- Simplify algebraic expressions (e.g., 2(x + 3) - 4(2x - 1))
- Solve systems of linear equations
- Find derivatives of functions (for calculus)
- Evaluate integrals (for calculus)
""")

# Link to the main application
if GROQ_API_KEY:
    st.header("Start Using the Tutor")
    st.write("Your API key is configured. You can now start using the Dual-Mode Math Tutor.")
    if st.button("Launch Math Tutor"):
        # Redirect to the main application
        st.switch_page("./integrated_math_tutor.py")
