import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    "features/segment_features.csv"
)

X = df.drop(
    "label",
    axis=1
)

y_text = df["label"]

encoder = LabelEncoder()

y = encoder.fit_transform(
    y_text
)

labels = encoder.classes_
# =========================
# Models
# =========================

models = {

    "SVM":
        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=10
                )
            )
        ]),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            objective="multi:softmax",
            num_class=len(labels),

            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,

            eval_metric="mlogloss",

            random_state=42
        )
}

# =========================
# Output Folder
# =========================

os.makedirs(
    "results/confusion_matrix",
    exist_ok=True
)

# =========================
# 5 Fold
# =========================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =========================
# Run
# =========================

for model_name, model in models.items():

    print("\n")
    print("=" * 50)
    print(model_name)
    print("=" * 50)

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in skf.split(X, y):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        y_true_all.extend(
            y_test
        )

        y_pred_all.extend(
            y_pred
        )

    # =====================
    # Classification Report
    # =====================

    report = classification_report(
        y_true_all,
        y_pred_all,
        target_names=labels,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_path = (
        f"results/{model_name}_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    print(
        f"Saved: {report_path}"
    )

    # =====================
    # Confusion Matrix
    # =====================

    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
    )

    plt.figure(
        figsize=(10,8)
    )

    plt.imshow(
        cm
    )

    plt.colorbar()

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45
    )

    plt.yticks(
        range(len(labels)),
        labels
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    # 顯示數字
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    save_path = (
        f"results/confusion_matrix/{model_name}_cm.png"
    )

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {save_path}"
    )

print("\n")
print("=" * 50)
print("Analysis Complete")
print("=" * 50)