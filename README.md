<p align="center">
  <img src="./assets/policywise.png" alt="PolicyWise Screenshot" width="85%">
</p>

<h1 align="center">🛡️ PolicyWise — AI Legal Policy Assistant</h1>

<p align="center">
An intelligent assistant that analyzes policy and legal clauses using RAG, a custom ML risk classifier, and LLM-powered explanations.
</p>

---

## ✨ What is PolicyWise?

PolicyWise is an AI-powered tool that helps Compliance and Legal teams quickly evaluate policy or legal clauses.

It combines:

- **RAG (Retrieval-Augmented Generation)** → Finds relevant text inside uploaded PDF policies  
- **Machine Learning Classifier** → Predicts if a clause is COMPLIANT or RISKY  
- **LLM Explanation (OpenAI)** → Gives clear explanations and safer rewrites  

This makes PolicyWise a smart internal assistant for reviewing documents.

---

## 🚀 Features

### 🔍 1. Document Search (RAG)
Upload PDF policy documents.  
PolicyWise will:

- Extract text  
- Break it into chunks  
- Create embeddings  
- Use FAISS to retrieve the most relevant sections

### 🛡️ 2. Risk Classifier (ML Model)
A Logistic Regression + TF-IDF classifier trained by me.  
It predicts:

- **COMPLIANT**  
- **RISKY**

With a confidence score.

### 🤖 3. AI Explanation (LLM-Enhanced)
If an OpenAI key is provided, PolicyWise can:

- Explain why a clause is risky  
- Highlight dangerous wording  
- Suggest a safer rewrite  
- Use RAG + ML to give better, more contextual answers  

---

## 📁 Project Structure

app.py                # Main Streamlit application
train_model.py        # Training script for the ML  
                       classifier
policy_model.pkl      # Saved classifier
policy_vectorizer.pkl # Saved TF-IDF vectorizer
requirements.txt
README.md



This project demonstrates:
-How RAG systems work
-How ML and LLMs can work together
-Ability to train and deploy a small classifier
-Practical features used in real companies
-Clear and simple user interface (Streamlit)
