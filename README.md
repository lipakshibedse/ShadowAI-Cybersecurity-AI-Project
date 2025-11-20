### 🛡️ AI-Cybersecurity ShadowAI Project  

# 🔐 ShadowAI – Cybersecurity Threat Detection System  
A machine-learning powered cybersecurity monitoring system designed to detect anomalies, suspicious user activity, and potential security threats in real time.

---

## 🚀 Live Demo (Streamlit Cloud)

Click below to try the deployed web app:

🔗 **https://shadowai-cybersecurity-ai-project-je6qlndnxnzhdifkygvnnq.streamlit.app/**

[![Streamlit App] (https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?style=for-the-badge&logo=streamlit)]
                    (https://shadowai-cybersecurity-ai-project-je6qlndnxnzhdifkygvnnq.streamlit.app/)

---

## 📌 About the Project
 ShadowAI is built to assist cybersecurity teams with **smart anomaly detection** using a trained machine-learning model.  
 It analyzes event logs, user behaviors, and system actions to determine whether an activity is:

- ✔ Normal  
- ⚠ Suspicious  
- ❌ Potential Threat  

The system uses a Random Forest Classifier trained on simulated cybersecurity logs.

---


#### ⚙️ Key Features  
✅ Real-time data visualization (AI usage patterns, departments, and activity heatmaps)  
✅ Machine Learning model (Random Forest) for predicting insider threats  
✅ Risk detection alerts (flagging high-risk employees or actions)  
✅ Integrated SQLite database (data stored locally for security)  
✅ Scalable, clean, and modular project structure 
✅ User-friendly Streamlit dashboard  


---


#### 🧠 Tech Stack  
| Layer | Technology Used |
|--------|----------------|
| **Frontend (UI)** | Streamlit (Python-based interactive web app) |
| **Backend (Logic)** | Python + ML Model (Random Forest) |
|**Machine Learning** | scikit-learn (Random Forest) |  
| **Database** | SQLite3 (stored as `shadowai.db`) |
| **Libraries** | pandas, numpy, matplotlib, scikit-learn, joblib |
|**Data Handling** | Pandas, NumPy | 
|**Model Storage** | Joblib | 

---

#### 🧩 Folder Structure
```
AI-Cybersecurity-ShadowAI-Project/
ShadowAI-Cybersecurity-AI-Project/
│
├── app/
│ └── app.py
│
├── data/
│ ├── generate_dataset.py
│ ├── shadowai.db
│ └── simulated_shadow_ai_logs.csv
│
├── models/
│ ├── rf_model.joblib
│ └── scaler.joblib
│
├── notebooks/
│ ├── 01_generate_dataset.ipynb
│ └── 02_model_training.ipynb
│
└── requirements.txt
└── README.md
```

---


---

## ⚙️ Installation (Run Locally)

Clone the repository:

```bash
git clone https://github.com/lipakshibedse/ShadowAI-Cybersecurity-AI-Project.git
cd ShadowAI-Cybersecurity-AI-Project


Install dependencies: pip install -r requirements.txt

Run the Streamlit app: streamlit run app/app.py

---



#### 🔒 Security Note  
All datasets and models are stored locally or in private repositories.  
No confidential company data is exposed publicly.





📊 Machine Learning Model

Random Forest Classifier
Trained using synthetic but realistic cybersecurity logs
Feature scaling applied using StandardScaler
Model and scaler stored in /models/ directory
Performs multi-class classification for:
Normal Activity
Suspicious Activity
Potential Threat





🚧 Future Scope

Deploy backend on cloud with API
Real-time log ingestion
Threat database integration
Deep learning-based threat detection
User behavior analytics




#### 👩‍💼 Developer  
**Lipakshi Bedse**  
MBA (AI & ML) | Cybersecurity & Data Analytics Enthusiast  
📧 [lipakshibedse20@gmail.com]  
📍 India  
## 🛡️ License
This project is for educational and research purposes only.


