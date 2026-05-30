import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# 讀取資料
df = pd.read_csv(
    "features/extracted_features.csv"
)

# X / y
X = df.drop("label", axis=1)
y = df["label"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Models
models = {
    "SVM": SVC(
        kernel="rbf",
        C=10
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Training & Evaluation
results = []

for name, model in models.items():

    print(f"Training {name}...")

    # 訓練
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)

    # accuracy
    acc = accuracy_score(
        y_test,
        y_pred
    )

    results.append([name, acc])

    print(f"{name} Accuracy: {acc:.4f}")
    print("-" * 40)

# Result Table
result_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy"]
)

print("\nFinal Result:")
print(result_df)