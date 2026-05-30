import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# =========================
# 三種資料集
# =========================

datasets = {
    "Global": "features/global_features.csv",
    "Middle": "features/middle_features.csv",
    "Segment": "features/segment_features.csv"
}

importance_tables = []

print("\n=========================")
print("Feature Importance Analysis")
print("=========================")

# =========================
# 分別計算 Importance
# =========================

for dataset_name, file_path in datasets.items():

    print(f"\nProcessing {dataset_name}...")

    df = pd.read_csv(file_path)

    feature_cols = [
        col for col in df.columns
        if col != "label"
    ]

    X = df[feature_cols]
    y = df["label"]

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    rf.fit(X, y)

    temp_df = pd.DataFrame({
        "Feature": feature_cols,
        dataset_name: rf.feature_importances_
    })

    importance_tables.append(temp_df)

# =========================
# 合併三個結果
# =========================

importance_df = importance_tables[0]

for table in importance_tables[1:]:

    importance_df = importance_df.merge(
        table,
        on="Feature"
    )

# =========================
# 平均重要度
# =========================

importance_df["Average"] = (
    importance_df["Global"]
    + importance_df["Middle"]
    + importance_df["Segment"]
) / 3

importance_df = importance_df.sort_values(
    by="Average",
    ascending=False
)

# =========================
# 顯示結果
# =========================

print("\n=========================")
print("Average Feature Importance")
print("=========================")

print(importance_df)

# =========================
# 存 CSV
# =========================

importance_df.to_csv(
    "features/feature_importance_summary.csv",
    index=False
)

print(
    "\nSaved: features/feature_importance_summary.csv"
)

# =========================
# Top Features
# =========================

print("\nTop 10 Features:")

print(
    importance_df[
        ["Feature", "Average"]
    ].head(10)
)

# =========================
# 畫圖
# =========================

plt.figure(figsize=(12,6))

plt.bar(
    importance_df["Feature"],
    importance_df["Average"]
)

plt.xticks(rotation=45)

plt.title(
    "Average Feature Importance\n(Global + Middle + Segment)"
)

plt.tight_layout()

plt.savefig(
    "features/feature_importance_summary.png"
)

print(
    "\nSaved: features/feature_importance_summary.png"
)