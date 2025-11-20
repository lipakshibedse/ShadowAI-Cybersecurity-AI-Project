### 🛡️ AI-Cybersecurity ShadowAI Project  

#### 📍 Overview  
This project — **ShadowAI** — is an intelligent AI-powered cybersecurity dashboard that detects insider data risks, monitors employee AI tool usage, and visualizes key metrics in real time.  
It integrates **Machine Learning**, **Streamlit Web App**, and **SQLite Database** to give security teams a powerful tool for identifying potential data leaks and high-risk activities.  

---

#### ⚙️ Features  
✅ Real-time data visualization (AI usage patterns, departments, and activity heatmaps)  
✅ Machine Learning model (Random Forest) for predicting insider threats  
✅ Risk detection alerts (flagging high-risk employees or actions)  
✅ Integrated SQLite database (data stored locally for security)  
✅ User-friendly Streamlit dashboard  

---

#### 🧠 Tech Stack  
| Layer | Technology Used |
|--------|----------------|
| **Frontend (UI)** | Streamlit (Python-based interactive web app) |
| **Backend (Logic)** | Python + ML Model (Random Forest) |
| **Database** | SQLite3 (stored as `shadowai.db`) |
| **Libraries** | pandas, numpy, matplotlib, scikit-learn, joblib |

---

#### 🧩 Folder Structure
```
AI-Cybersecurity-ShadowAI-Project/
│
├── app/
│   └── app.py
│
├── data/
│   ├── ai_logs.csv
│   └── shadowai.db
│
├── models/
│   └── rf_model.joblib
│
├── requirements.txt
└── README.md
```

---

#### 🚀 How to Run (Locally)
```bash
cd AI-Cybersecurity-ShadowAI-Project/app
streamlit run app.py
```
Then open: http://localhost:8501 in your browser.

---

#### 🌍 How to Deploy (Online)
1. Upload the project on **GitHub**  
2. Go to [Streamlit Cloud](https://share.streamlit.io)  
3. Sign in with GitHub  
4. Select your repo and deploy  
5. Done 🎉 Your app will be live at:
   ```
   https://shadowai-cybersecurity-ai-project-je6qlndnxnzhdifkygvnnq.streamlit.app/
   ```

---

#### 🔒 Security Note  
All datasets and models are stored locally or in private repositories.  
No confidential company data is exposed publicly.

---

#### 👩‍💼 Developer  
**Lipakshi Bedse**  
MBA (AI & ML) | Cybersecurity & Data Analytics Enthusiast  
📧 [lipakshibedse20@gmail.com]  
📍 India  
## 🛡️ License
This project is for educational and research purposes only.
