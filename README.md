# 🤖 Intelligent Chat Analysis System using AI/ML

An AI-powered web application that analyzes WhatsApp chat data and generates meaningful insights using Data Analysis, Machine Learning, and Natural Language Processing (NLP).

---

## 📌 Project Overview

This project takes exported WhatsApp chat files (.txt) and converts raw unstructured data into useful insights such as activity patterns, sentiment, emotions, topics, and summaries.

It provides an interactive dashboard where users can easily explore chat data visually.

---

## 🔥 Key Features

### 📊 Chat Analysis
- Total messages, words, media, links
- Most active users
- Daily & monthly activity timeline

### 📈 Visualization
- WordCloud of most used words
- Emoji analysis
- Heatmaps and charts

### 🧠 AI/ML Features
- Sentiment Analysis (Positive/Negative)
- Emotion Detection (Joy, Anger, Sadness, etc.)
- Topic Modeling (LDA)
- Chat Summarization
- User Behavior Profiling

### 💡 Advanced
- Auto Insight Generation
- PDF Report Download

---

## 🛠️ Tech Stack

- **Python** (Core Language)
- **Pandas, NumPy** (Data Processing)
- **Matplotlib, Seaborn** (Visualization)
- **Scikit-learn** (Machine Learning)
- **Hugging Face Transformers** (NLP Models)
- **Streamlit** (Web App UI)
- **ReportLab** (PDF Generation)

---

## 📂 Project Structure
chat-analysis-system/
│
├── app.py
├── preprocessor.py
├── helper.py
├── ai_features.py
├── insights.py
├── report_utils.py
├── stop_hinglish.txt
├── requirements.txt


---

## ⚙️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/chat-analysis-system.git
cd chat-analysis-system
2. Install Dependencies
pip install -r requirements.txt
3. Run Application
streamlit run app.py
📥 How to Use
Export WhatsApp chat as .txt
Upload file in the app
Select user or overall chat
Click Show Analysis
Explore insights and download PDF report
⚠️ Limitations
Supports only .txt files
English language only
Max file size: 200MB
🔮 Future Scope
Real-time chat analysis
Multilingual support (Hindi, Hinglish)
Advanced deep learning models
Full-stack deployment (Web/Mobile)
