# IdleVision-AI 🧠🎥
### AI-Driven Employee Productivity & Idle Detection System

**IdleVision-AI** is an AI-powered **computer vision system** that monitors employee activity through live video analysis.  
It detects idle or inactive behavior in real time and provides **automated productivity insights** via a Streamlit dashboard.

---

## 🚀 Features
- 🔍 Real-time activity detection using **MediaPipe Pose Estimation**
- 📊 Live analytics dashboard built with **Streamlit + Plotly**
- ⏱️ Logs live data every second (`live_status_log.csv`)
- 🕒 Records idle events for HR reports (`idle_log.csv`)
- ⚙️ Fully configurable thresholds and camera sources
- 🌐 Deployable locally or on Streamlit Cloud

---

## 🧰 Tech Stack
| Category | Tools |
|-----------|--------|
| Programming | Python 3.10+ |
| Computer Vision | OpenCV, MediaPipe |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, NumPy |
| ML (Optional) | TensorFlow / CNN |
| Deployment | Streamlit Cloud / Localhost |

---

## 📊 Project Overview
IdleVision-AI enables organizations to:
- Automatically monitor work efficiency  
- Detect idle periods and generate monthly reports  
- Enhance employee engagement and performance transparency  

---

## 🧠 Architecture
```text
[ Camera Feed ] 
       ↓
[ Pose Detection (MediaPipe) ]
       ↓
[ Movement Analysis ]
       ↓
[ Idle Detection Logic ]
       ↓
[ Live Data Logging ] ---> live_status_log.csv
[ Idle Event Logger ] ---> idle_log.csv
       ↓
[ Streamlit Dashboard Visualization ]
