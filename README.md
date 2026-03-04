# 🎓 Student Performance Predictor

> An ML-powered web application that predicts a student's final percentage using an **AdaBoost Regressor** model — built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-10b981?style=flat-square)

---

## 📸 Preview

```
┌─────────────────────────────────────────────────────┐
│         🎓  Student Performance Predictor           │
│   AdaBoost Regressor  ·  ML Model                   │
├─────────────────────────────────────────────────────┤
│  Academic Factors                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │Study Hrs │  │ Failures │  │Absences  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│  Personal Details                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │Social Hrs│  │ Gender   │  │ Internet │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│         [ Predict Final Percentage → ]              │
├─────────────────────────────────────────────────────┤
│            Predicted Final Percentage               │
│                    87.43 %                          │
│         🌟 Excellent · Grade A+                     │
│  ████████████████████████░░░░░░  Study: 6h/day      │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **AdaBoost Regressor** with a Decision Tree base estimator (400 estimators)
- Clean, dark-themed UI with smooth CSS animations and gradient accents
- Real-time grade classification (O / A+ / A / B / C / F)
- Animated progress bar showing the predicted score visually
- Responsive 3-column layout for inputs
- Cached model training via `@st.cache_resource` for fast reloads

---

## 🧠 Model Details

| Parameter       | Value                     |
|-----------------|---------------------------|
| Algorithm       | AdaBoost Regressor        |
| Base Estimator  | Decision Tree (max_depth=4)|
| n_estimators    | 400                       |
| learning_rate   | 0.1                       |
| Test Size       | 20%                       |
| Scaler          | StandardScaler            |

---

## 📁 Project Structure

```
student-performance-predictor/
├── Student_Performance_Dataset.csv   # Training dataset
├── stream.py                             # Main Streamlit application
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Sanjaymo/Student-Performance-Prediction.git
cd student-performance-predictor
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place your CSV file at:
```
dataset/Student_Performance_Dataset.csv
```

The CSV must contain these columns:

| Column             | Type    | Description                        |
|--------------------|---------|------------------------------------|
| `Study_Hours`      | int     | Daily study hours (0–12)           |
| `Failures`         | int     | Number of past failures            |
| `Absences`         | int     | Total absences                     |
| `Social_Media`     | int     | Daily social media hours           |
| `Gender`           | string  | Male / Female                      |
| `Internet`         | string  | Yes / No                           |
| `Final_Percentage` | float   | **Target variable** (0–100)        |

### 5. Run the App

```bash
streamlit run stream.py
```

Open your browser at **http://localhost:8501**

---

## 📦 requirements.txt

```
streamlit>=1.28.0
pandas>=1.5.0
scikit-learn>=1.2.0
```

> Generate it yourself anytime:
> ```bash
> pip freeze > requirements.txt
> ```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repository
4. Set **Main file path** to `stream.py`
5. Click **Deploy** 🚀

> **Note:** Make sure your dataset CSV is committed to the repository, or host it externally and update the path in `stream.py`.

---

## 🎨 UI Highlights

- **Font:** Sora (display) + JetBrains Mono (labels/badge)
- **Theme:** Deep dark background with blue-violet gradient accents
- **Animations:** Fade-in hero, card slide-up, pop-in result, animated progress bar
- **Color Palette:**
  - Background: `#0b0f1a`
  - Accent Blue: `#4f8ef7`
  - Accent Purple: `#7c3aed`
  - Success Green: `#10b981`

---

## 👤 Author

**Sanjay Choudhari**

- 📧 [sanjaychoudhari288@gmail.com](mailto:sanjaychoudhari288@gmail.com)
- 📞 +91 9963785768
- 🐙 [github.com/SanjayChoudhari](https://github.com/Sanjaymo)

---

## 📄 License

This project is licensed under the **Apache License** — feel free to use, modify, and distribute.

---

<p align="center">
  Made with ❤️ using Python & Streamlit
</p>
