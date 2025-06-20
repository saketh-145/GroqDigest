# GroqDigest

---

````markdown
# 📰 GroqDigest: News Summarizer + Q&A App

An AI-powered app that summarizes news articles and answers user questions in multiple languages using the Groq API and Llama 3.

![Screenshot](./screenshot.png) <!-- Add your screenshot image path -->

---

## 🚀 Features

- 🌐 **Multilingual Summarization** (English, Hindi, Telugu, Spanish, French, German)
- ❓ **Question Answering** from article content
- 🔗 **URL-based scraping** using `newspaper3k`
- 🧠 **FAISS Vector Store** for retrieval-based QA
- ⚙️ **Groq API** with Llama 3 integration
- 🖥️ Simple **Streamlit UI**

---

## 🧰 Tech Stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Groq API](https://console.groq.com/)
- [newspaper3k](https://github.com/codelucas/newspaper)

---

## 🛠️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/GroqDigest.git
cd GroqDigest
````

2. **Create a virtual environment:**

```bash
python -m venv venv
# Activate it:
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure `.env`:**

Create a `.env` file in the root folder with:

```env
OPENAI_API_KEY=your_openrouter_or_groq_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
```

5. **Run the Streamlit app:**

```bash
streamlit run app.py
```

---

## 📸 UI Preview

| Summarization                     | Q\&A Interface          |
| --------------------------------- | ----------------------- |
| ![Summary](./preview-summary.png) | ![QA](./preview-qa.png) |

---

## 📂 Project Structure

```bash
GroqDigest/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── venv/
```

---

## 💡 Future Improvements

* Upload PDFs or text files for summarization
* Save Q\&A history
* Add audio/video summary output
* Support for more LLM providers (e.g., Claude, Gemini)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT License – see `LICENSE` file for details.

```

---

Let me know if you want this customized further (e.g., for a team, with badges, or deploy instructions).
```
