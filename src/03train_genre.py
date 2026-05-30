import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 讀取 CSV
df = pd.read_csv(
    "features/extracted_features.csv"
)

# X 與 y
X = df.drop("label", axis=1)

y = df["label"]

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 建立 SVM 模型
model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

# 訓練
print("Training SVM...")
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# Classification Report
print("\\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Confusion Matrix
print("\\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)