# **Direct Preference Optimization (DPO) Model Training**

## **Overview**
This repository contains the implementation of **Direct Preference Optimization (DPO)** for fine-tuning **GPT-2** using a preference-based dataset. The model has been trained and uploaded to **Hugging Face** and is accessible for inference via a **Streamlit web application**.

## **Dataset Selection & Preprocessing**
- **Dataset Used**: [`Dahoas/static-hh`](https://huggingface.co/datasets/Dahoas/static-hh)
- **Source**: Hugging Face Dataset Hub
- **Preprocessing Steps**:
  - Extracted relevant fields (`prompt`, `chosen`, `rejected`).
  - Formatted inputs to maintain consistency in response generation.
  - Cleaned and structured the data to ensure suitability for preference-based training.

---

## **Training the Model with DPOTrainer**
- The **GPT-2** model was fine-tuned using **Direct Preference Optimization (DPO)**.
- Hyperparameters were optimized using multiple combinations.
- The training loop used `DPOTrainer` from Hugging Face's `trl` library.

### **Hyperparameters Used**
| Hyperparameter  | Value |
|----------------|-------|
| Learning Rate  | 1e-3  |
| Batch Size     | 8     |
| Epochs         | 5     |
| Beta           | 0.1   |

- The model was trained for **5 epochs**, and the **best model was selected based on the lowest evaluation loss**.

---

## **Pushing the Model to Hugging Face**
- The trained model and tokenizer were uploaded to **Hugging Face Hub**.
- **Model Repository**: [PhuePwint/dpo_gpt2](https://huggingface.co/PhuePwint/dpo_gpt2)
- **Best Model Path**: `checkpoint-4`

---

## **Web Application**
### **Demo Application**
A **Streamlit-based web application** was developed to demonstrate the trained model's capabilities.

### **How It Works**
- Users enter a prompt in the application.
- The fine-tuned GPT-2 model generates a response.
- The application displays the response in real-time.

### **Run the Application**
To run the application locally:
```bash
streamlit run app.py
