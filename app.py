import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model from Hugging Face (Corrected Path)
HF_MODEL_REPO = "PhuePwint/dpo_gpt2"

@st.cache_resource
def load_model():
    """Loads the fine-tuned GPT-2 model and tokenizer from Hugging Face"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL_REPO)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Ensure tokenizer has a pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model and tokenizer
tokenizer, model, device = load_model()

# Function to generate response
def generate_response(prompt, max_tokens=100):
    """Generates a response using the fine-tuned GPT-2 model"""
    if model is None or tokenizer is None:
        return "Error: Model not loaded."

    # Format prompt
    formatted_prompt = f"Q: {prompt}\nA:"

    # Tokenize input
    input_ids = tokenizer(
        formatted_prompt, return_tensors="pt", padding=True, truncation=True
    ).input_ids.to(device)

    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.5,  # Adjust randomness
            top_p=0.85,  # Balance nucleus sampling
            top_k=50,  # Increase diversity
            repetition_penalty=1.4,  # Reduce repetitive responses
            do_sample=True,  # Enable diverse responses
            eos_token_id=tokenizer.eos_token_id,  # Ensure proper stopping
            pad_token_id=tokenizer.pad_token_id,  # Avoid padding issues
        )

    # Decode and clean response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    
    # Ensure we don't return the exact prompt as response
    if response.lower() == prompt.lower():
        return "I'm not sure, but I can try to provide more details!"

    return response

# Streamlit Web App UI
st.title("📝 GPT-2 DPO Model Demo")
st.write("This app allows you to interact with the fine-tuned GPT-2 model uploaded to Hugging Face.")

# User input
user_input = st.text_area("Enter a prompt:", "Who is the president of the USA?")

# Generate response on button click
if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        response = generate_response(user_input)
        st.subheader("Response:")
        st.write(response)

# Footer
st.markdown("---")
st.write("📌 *Fine-tuned GPT-2 model deployed using Streamlit & Hugging Face*")
st.write("👩‍💻 *Developed by [Phue Pwint Thwe](https://github.com/phuepwint-thwe)*")