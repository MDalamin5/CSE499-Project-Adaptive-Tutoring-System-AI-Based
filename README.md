# CSE499 Project: AI-Based Adaptive Algebra Tutor

## Overview

This project is an AI-powered adaptive tutoring system designed to help students learn algebra. It leverages the Phi-3 Mini 4k Instruct model, fine-tuned with a carefully crafted dataset, to provide personalized and interactive learning experiences. The system is built using Streamlit for the user interface and is intended to be a capstone project demonstrating advanced AI techniques in education.

## Key Features

*   **Interactive Chat-Based Tutoring:** Provides a conversational learning experience where students can ask questions and receive step-by-step guidance.
*   **Adaptive Learning:** The system aims to adapt to the student's individual learning style and skill level (future implementation).
*   **Personalized Feedback:** Offers tailored feedback and hints based on the student's responses (future implementation).
*   **Step-by-Step Solutions:** Provides detailed solutions to algebra problems, explaining the reasoning behind each step.
*   **Clear and Intuitive UI:** Uses Streamlit to create a user-friendly and engaging interface.
*   **Leverages Hugging Face Ecosystem:** Utilizes the Phi-3 Mini model and associated tools from the Hugging Face Transformers library.

## Technologies Used

*   **Large Language Model (LLM):** Microsoft Phi-3 Mini 4k Instruct
*   **Fine-Tuning:** Parameter-Efficient Fine-Tuning (PEFT)
*   **Frameworks:** PyTorch, Transformers, Streamlit, PEFT
*   **Libraries:** `transformers`, `datasets`, `streamlit`, `peft`
*   **Hugging Face Hub:** For model storage and sharing
*   **(Future) LangChain:** For advanced reasoning, tool integration, and memory management
*   **(Future) LangGraph:** For personalized learning path management
*   **(Future) RAG (Retrieval-Augmented Generation):** To build in formula, code explanation

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/MDalamin5/CSE499-Project-Adaptive-Tutoring-System-AI-Based.git
    cd CSE499-Project-Adaptive-Tutoring-System-AI-Based
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    *   **Note:** A `requirements.txt` file should be added to your repository listing all the dependencies.  Example content:

        ```
        torch
        transformers
        streamlit
        peft
        accelerate
        bitsandbytes
        huggingface_hub
        ```

4.  **Set up your Hugging Face token:**

    *   Log in to your Hugging Face account:

        ```bash
        huggingface-cli login
        ```

    *   You'll need to accept terms of service for the base model
        go to https://huggingface.co/microsoft/Phi-3-mini-4k-instruct accept ToS

## Usage

1.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

2.  **Interact with the tutor:** The app will open in your web browser, allowing you to start a conversation with the algebra tutor.

## Project Structure

*   `app.py`: The main Streamlit application file.
*   `LoRA-FineTuning/`: Directory containing any locally saved model files or configuration.  If you are only loading from the Hub, this may not exist.
*   `Data-multi-turn-instruction/`: Directory containing the training dataset.
*   `notebooks/`: Directory containing any Jupyter notebooks used for data exploration or model development.
*   `requirements.txt`: Lists the Python packages required to run the project.
*   `README.md`: This file.

## Model Fine-Tuning

The project utilizes the Phi-3 Mini 4k Instruct model, fine-tuned for algebra tutoring. The fine-tuning process involved:

1.  **Data Preparation:** A dataset of 250 different algebra problems was created.  Each problem was augmented with 5 different variants to simulate realistic student interactions and address potential misunderstandings. The variants were created using the approach below:

    * Variant 1 (Direct understanding): The student understands the first explanation clearly without needing further clarification.
    * Variant 2 (Misinterpretation): The student misunderstands the operation (e.g., adds instead of subtracting) and asks why subtraction is needed.
    * Variant 3 (Conceptual difficulty): The student doesn’t understand the concept (e.g., isolating variables) and needs a detailed explanation.
    * Variant 4 (Repetition request): The student asks for the explanation to be repeated but doesn’t need a simplified version.
    * Variant 5 (Topic diversion): The student asks a related question (e.g., “Why do we subtract instead of divide here?”), and the system briefly explains before continuing.

    Different Phrasings for Student Responses
        Ensure the student interactions vary in how they ask for help or provide feedback
    Simulate Varying System Teaching Styles
        The system’s explanation can be rephrased or adapted in each variant
    Add Realistic Context for Problem-Solving
        Make the problem more engaging by embedding it into relatable scenarios:
    Diversify Feedback and Teaching Flow
        Simulate varying conversational flows in each variant

2.  **Training:** The model was fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) to adapt it to the specific task while minimizing resource requirements.
3.  **Model Repository:** The fine-tuned model is available on the Hugging Face Hub: [alam1n/phi3-mini-algebra-tutor-v4](https://huggingface.co/alam1n/phi3-mini-algebra-tutor-v4)

## Future Enhancements

The project is still under development. Planned enhancements include:

*   **Integration with LangChain:**
    *   Implementing more sophisticated reasoning and problem-solving capabilities.
    *   Integrating tools like `sympy` or Wolfram Alpha for symbolic calculations.
    *   Adding memory to track conversation history and student progress.
*   **Personalized Learning with LangGraph:**
    *   Creating student profiles and tailoring the learning path accordingly.
    *   Dynamically adjusting the difficulty of problems based on student performance.
    *   Providing personalized feedback and hints.
*   **Retrieval-Augmented Generation (RAG):**
    *   Building a database of algebra formulas, theorems, and definitions.
    *   Retrieving relevant information from the database to provide context for the model's responses.
*   **Improved UI/UX:**
    *   Adding more interactive elements to the user interface.
    *   Implementing visualizations to illustrate mathematical concepts.
    *   Improving the overall design and user experience.
*   **Gamification:**
    *   Introducing elements of gamification, such as points, badges, and leaderboards, to motivate students and make learning more fun.
*   **Deployment:**
    *   Deploying the app to a cloud platform to make it accessible to a wider audience.
    *   **GPRO training:** This will significantly improve the model's ability to generate high-quality, relevant, and engaging responses.

## Challenges and Lessons Learned

*   Fine-tuning large language models requires careful data preparation and experimentation with training parameters.
*   Maintaining a balance between model accuracy and creativity is a key challenge.
*   Creating a user-friendly and engaging interface is essential for a successful tutoring system.

## Team Members

*   [Md Al Amin](https://github.com/MDalamin5)
*   [Aminul Islam](https://github.com/AmiulIslam1729)
*   [Mohaimen Azad](https://github.com/niloy0912)

## License

[MIT License](https://github.com/MDalamin5/CSE499-Project-Adaptive-Tutoring-System-AI-Based?tab=MIT-1-ov-file)

## Acknowledgments

*   Hugging Face for providing the Transformers library and the Hub for model sharing.
*   The open-source community for their contributions to the field of natural language processing.