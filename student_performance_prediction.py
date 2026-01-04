# 1. Import Required Libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# 2. Load CSV Dataset
df = pd.read_csv("student_performance_real_life (2).csv")

print(df.head())


# 3. Features (X) and Target (y)
X = df[["Hours_Studied", "Attendance", "Marks"]]
y = df["Result"]   # 0 = Fail, 1 = Pass


# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model Selection
model = LogisticRegression()

# 6. Train the Model
model.fit(X_train, y_train)


# 7. Model Prediction
y_pred = model.predict(X_test)


# 8. Model Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 9. USER INPUT PREDICTION
print("\n--- Student Performance Prediction ---")

hours = float(input("Enter hours studied per day: "))
attendance = float(input("Enter attendance percentage: "))
marks = float(input("Enter expected marks: "))

input_data=pd.DataFrame([[hours, attendance, marks]],
                        columns=["Hours_Studied", "Attendance", "Marks"])

result = model.predict(input_data)

if result[0] == 1:
    print("Prediction: Student will PASS ")
else:
    print("Prediction: Student will FAIL ")






































