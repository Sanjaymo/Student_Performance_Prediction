import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# 1. Load Data
df = pd.read_csv("Student_Performance_Dataset.csv")

# 2. Prevent Data Leakage! 
# Drop ID, direct scores, and pass/fail labels so the model relies on habits/demographics
cols_to_drop = ['Student_ID', 'Math_Score', 'Science_Score', 'English_Score', 
                'Performance_Level', 'Pass_Fail', 'Final_Percentage']

X = df.drop(columns=cols_to_drop)
y = df["Final_Percentage"]

# 3. Encode Categorical Variables
cat_cols = X.select_dtypes(include=["object", "category"]).columns
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 4. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Train Model
dt = DecisionTreeRegressor(max_depth=6) # Increased depth slightly for better learning
model = AdaBoostRegressor(estimator=dt, n_estimators=400, learning_rate=0.05, random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")
print("\n--- Enter Student Details ---")

# 6. Ask for the EXACT columns the model was trained on
age = int(input("Age (e.g., 15): "))
gender = input("Gender (Male/Female): ")
student_class = int(input("Class (e.g., 10): "))
study_hours = float(input("Study hours per day (e.g., 4.5): "))
attendance = float(input("Attendance Percentage (e.g., 85): "))
parent_edu = input("Parental Education (High School/Graduate/Postgraduate): ")
internet = input("Internet access (Yes/No): ")
extracurricular = input("Extracurricular Activities (Yes/No): ")
prev_score = float(input("Previous Year Score (e.g., 75): "))

# 7. Create DataFrame using the exact original column names
user_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Class": [student_class],
    "Study_Hours_Per_Day": [study_hours],
    "Attendance_Percentage": [attendance],
    "Parental_Education": [parent_edu],
    "Internet_Access": [internet],
    "Extracurricular_Activities": [extracurricular],
    "Previous_Year_Score": [prev_score]
})

# 8. Encode the user input
user_encoded = pd.get_dummies(user_data)

# Reindex to match the training columns, filling ANY missing dummy variables with 0
user_encoded = user_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# 9. Scale and Predict
user_scaled = scaler.transform(user_encoded)
prediction = model.predict(user_scaled)

print(f"\nPredicted Final Percentage: {round(prediction[0], 2)}%")
