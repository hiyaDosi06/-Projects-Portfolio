# 🚀 Machine Learning, Deep Learning & GenAI Projects Portfolio

Welcome to my project portfolio repository! This collection contains 9 practical projects spanning **Supervised Learning**, **Computer Vision (CNNs)**, **Reinforcement Learning (DQN)**, **Recommender Systems**, **Cloud Deployment**, and **Generative AI / RAG Architecture**.

---

## 📂 Project Directory Overview

| # | Project Title | Core Tech Stack | Description |
| :---: | :--- | :--- | :--- |
| **01** | **Adult Census Income Classification** | `Scikit-Learn`, `Pandas`, `XGBoost` | Binary classification model predicting whether individual income exceeds $50K/yr based on census data. |
| **02** | **CIFAR-10 Image Classification** | `PyTorch`, `CNN`, `Torchvision` | Convolutional Neural Network built from scratch with BatchNorm & Dropout for multi-class image recognition. |
| **03** | **LFW Wild Face Recognition** | `PyTorch`, `ResNet-18`, `OpenCV` | Transfer learning pipeline trained on the Labeled Faces in the Wild (LFW) dataset for face identification. |
| **04** | **Brain MRI Cancer Detection** | `PyTorch`, `ResNet-50`, `PIL` | Binary deep learning classifier fine-tuned to distinguish cancerous vs. non-cancerous brain MRI scans. |
| **05** | **Cart-Pole RL Agent Training** | `Gymnasium`, `PyTorch` | Deep Q-Network (DQN) reinforcement learning agent trained to balance a pole on a moving cart. |
| **06** | **Lunar Lander RL Agent Training** | `Gymnasium`, `PyTorch` | DQN agent with experience replay and target networks trained to land a spacecraft safely on a landing pad. |
| **07** | **Movie Recommendation System** | `Scikit-Learn`, `SciPy`, `Pandas` | Hybrid recommendation engine combining Content-Based Filtering (TF-IDF) and Collaborative Filtering (SVD). |
| **08** | **End-to-End Render Deployment** | `FastAPI`, `Uvicorn`, `Render` | Production-ready RESTful web API deployed on Render cloud platform using structured build scripts. |
| **09** | **RAG Chatbot (Capstone Project)** 🌟 | `LangChain`, `FAISS`, `Streamlit` | Enterprise Retrieval-Augmented Generation (RAG) system querying custom PDF documents with source attribution. |

---

## 🛠️ Project Summaries

### 01. Adult Census Income Classification
* **Objective:** Predict individual income levels based on demographic parameters (education, occupation, age, capital gain).
* **Key Steps:** Categorical encoding, missing value imputation, feature scaling, and model evaluation using ROC-AUC and F1-Score.

### 02. CIFAR-10 Image Classification using CNN
* **Objective:** Classify $32\times32$ color images into 10 distinct object categories.
* **Key Steps:** Custom CNN architecture with alternating Conv2D, MaxPool2D, BatchNorm, and Dropout layers to prevent overfitting.

### 03. Face Recognition in the Wild (LFW Dataset)
* **Objective:** Perform robust facial identification under unconstrained conditions (lighting, pose, expression).
* **Key Steps:** Data preprocessing, facial alignment, transfer learning via pre-trained ResNet-18, and classification fine-tuning.

### 04. Cancer Detection using MRI Images
* **Objective:** Detect brain tumors/cancerous tissue from medical MRI scans.
* **Key Steps:** Spatial image augmentations, fine-tuning ResNet-50 head, BCE with Logits Loss optimization, and evaluation metrics (Precision, Recall, Confusion Matrix).

### 05. Cart-Pole RL Agent Training
* **Objective:** Solve the continuous control inverted pendulum task (`CartPole-v1`).
* **Key Steps:** Implemented Deep Q-Learning (DQN) with an $\epsilon$-greedy exploration policy and Experience Replay Memory buffer.

### 06. Lunar Lander RL Agent Training
* **Objective:** Train a lander module to control its thrusters and land safely inside a designated landing pad (`LunarLander-v3`).
* **Key Steps:** Deep Q-Network with separate Policy and Target networks, Huber Loss minimization, and target network synchronization.

### 07. Movie Recommendation System
* **Objective:** Suggest relevant movies based on user viewing history and metadata.
* **Key Steps:** TF-IDF Vectorization over plot keywords (Content-Based) and Singular Value Decomposition (SVD) matrix factorization (Collaborative Filtering).

### 08. End-to-End Render Deployment
* **Objective:** Deploy a Python machine learning backend to a cloud hosting environment.
* **Key Steps:** FastAPI RESTful endpoints, CORS configuration, environment variable management, and automated Git-triggered deployment via Render Blueprints.

### 09. 🌟 Capstone Project: RAG Chatbot
* **Objective:** Build an interactive Q&A assistant capable of answering natural language questions strictly grounded in custom PDF documents.
* **Key Steps:** 
  * Document Ingestion using `PyPDFLoader`
  * Text Chunking via `RecursiveCharacterTextSplitter` (1000 size, 200 overlap)
  * Vector Indexing using HuggingFace Embeddings (`all-MiniLM-L6-v2`) and **FAISS** vector database
  * Conversational QA interface using **Streamlit** and **LangChain** with document source attribution.

