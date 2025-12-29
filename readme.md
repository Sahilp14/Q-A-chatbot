# Q-A Chatbot — AI-Powered Conversational Assistant

A professional, modular Question-Answering chatbot built using **Streamlit** and **LangChain**, supporting multiple Large Language Models including **Google Gemini**, **OpenAI**, and **Ollama (local models)**.

This project allows users to interact with an AI assistant through a clean web interface, select different models, and customize response behavior using adjustable parameters.

---

## 🚀 Features

- 💬 **Multi-Model Support**
  - Google Gemini
  - OpenAI (GPT models)
  - Ollama (local LLMs)

- 🎛️ **Configurable Parameters**
  - Temperature control
  - Max token limit

- 🖥️ **Interactive UI**
  - Built using Streamlit
  - Simple and responsive interface

- 🧩 **Extensible Architecture**
  - Easy to add RAG, embeddings, vector databases, or memory

---

## 📁 Project Structure

Q-A-chatbot/
│
├── google/ # Google Gemini integration
├── openai/ # OpenAI GPT integration
├── ollama/ # Ollama local model integration
├── requirements.txt # Project dependencies
├── .gitignore
└── README.md

| Technology     | Purpose |
|----------------|---------|
| Python         | Core programming language |
| Streamlit     | Web UI |
| LangChain     | LLM orchestration |
| OpenAI API    | Cloud-based LLM |
| Google Gemini | Generative AI |
| Ollama        | Local LLM inference |

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sahilp14/Q-A-chatbot.git
cd Q-A-chatbot