# 📦 Supply Chain Intelligence Hub

An AI-powered supply chain analytics dashboard built with Python, Streamlit, and Llama 3.3 70B — combining Machine Learning anomaly detection with Lean Six Sigma DMAIC root cause analysis.

---

## 🚀 Live Demo
**[👉 Click here to try the live app](https://sc-ai-dashboard.streamlit.app)**
> Upload any supply chain CSV and get instant AI-powered insights

---

## ✨ Features

- **KPI Dashboard** — Real-time metrics: revenue, lead time, defect rate, units sold
- **Interactive Charts** — Revenue by product, lead time vs defect rate, shipping performance, transportation mix
- **Supplier Scorecard** — Rank suppliers by defect rate, lead time, and revenue with 🔴🟡🟢 risk flags
- **AI Anomaly Detection** — Isolation Forest ML model flags outlier SKUs and suppliers automatically
- **AI Root Cause Analysis** — Describe any SC issue → get a full DMAIC structured analysis powered by Llama 3.3 70B
- **Upload Any CSV** — Works with any supply chain dataset, not just the default

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Data Processing | Python, Pandas |
| ML Model | Scikit-learn (Isolation Forest) |
| Visualizations | Plotly Express |
| AI Engine | Llama 3.3 70B via Groq API |
| Framework | DMAIC (Lean Six Sigma) |

---

## 📊 Dataset

Uses the [Supply Chain Analysis Dataset](https://www.kaggle.com/datasets/amirmotefaker/supply-chain-dataset) from Kaggle — 100 SKUs across 3 product lines (haircare, skincare, cosmetics) with supplier, logistics, and quality metrics.

---

## ⚙️ Setup

```bash
# Clone the repo
git clone https://github.com/Sai-Gangawane/sc-ai-dashboard.git
cd sc-ai-dashboard

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

---

## 🔑 Environment Variables

Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

---

## 👤 Author

**Sai Santosh Gangawane**
MS Engineering Management — Syracuse University
Lean Six Sigma Black Belt (ICBB) | CSCMP Certified

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/sai-gangawane)
