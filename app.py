import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

df = pd.read_csv("Student_Performance_Dataset.csv")
X = df.drop("Final_Percentage", axis=1)
y = df["Final_Percentage"]

cat_cols = X.select_dtypes(include=["object", "category"]).columns
X_encoded = pd.get_dummies(X,columns=cat_cols,drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

dt = DecisionTreeRegressor(max_depth=4)
model = AdaBoostRegressor(estimator=dt,n_estimators=400,learning_rate=0.1,random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")
print("\nEnter Student Details:\n")

study_hours = float(input("Study hours per day: "))
failures = int(input("Number of past failures: "))
absences = int(input("Number of absences: "))
social_media = float(input("Social media hours per day: "))
gender = input("Gender (Male/Female): ")
internet = input("Internet access (Yes/No): ")

user_data = pd.DataFrame({
    "Study_Hours":[study_hours],
    "Failures":[failures],
    "Absences":[absences],
    "Social_Media":[social_media],
    "Gender":[gender],
    "Internet":[internet]
})
user_encoded = pd.get_dummies(user_data)
user_encoded = user_encoded.reindex(columns=X_encoded.columns,fill_value=0)

user_scaled = scaler.transform(user_encoded)
prediction = model.predict(user_scaled)
print("\nPredicted Final Percentage:", round(prediction[0],2))