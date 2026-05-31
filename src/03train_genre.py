import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier

# =========================
# Load Feature Importance
# =========================

importance_df = pd.read_csv(
    "features/feature_importance_summary.csv"
)

ranking = importance_df["Feature"].tolist()

# =========================
# Feature Sets
# =========================

feature_sets = {
    "Top3": ranking[:3],
    "Top5": ranking[:5],
    "Top8": ranking[:8],
    "Top10": ranking[:10]
}

# =========================
# Datasets
# =========================

datasets = {
    "Global": "features/global_features.csv",
    "Middle": "features/middle_features.csv",
    "Segment": "features/segment_features.csv"
}

# =========================
# Models
# =========================
# =========================
# GMM Classifier
# =========================

def gmm_classifier(
    X_train,
    y_train,
    X_test,
    n_components=4
):

    labels = np.unique(y_train)

    gmms = {}

    # 每個類別建立一個 GMM
    for label in labels:

        class_data = X_train[
            y_train == label
        ]

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=42
        )

        gmm.fit(class_data)

        gmms[label] = gmm

    predictions = []

    # 預測
    for sample in X_test:

        scores = {}

        for label in labels:

            score = gmms[label].score_samples(
                sample.reshape(1, -1)
            )[0]

            scores[label] = score

        pred = max(
            scores,
            key=scores.get
        )

        predictions.append(pred)

    return np.array(predictions)
models = {

    "SVM":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                C=10
            ))
        ]),

    "KNN":
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=5
            ))
        ]),

    "RandomForest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            objective="multi:softmax",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42
        )
}

# =========================
# Result Storage
# =========================

results = []

# =========================
# Main Experiment
# =========================

for dataset_name, dataset_path in datasets.items():

    print("\n")
    print("=" * 60)
    print(dataset_name)
    print("=" * 60)

    df = pd.read_csv(dataset_path)

    # All Features
    all_features = [
        col
        for col in df.columns
        if col != "label"
    ]

    feature_sets["All"] = all_features

    y_text = df["label"]

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        y_text
    )

    # =========================
    # Feature Sets
    # =========================

    for feature_name, feature_columns in feature_sets.items():

        X = df[feature_columns]

        print(
            f"\nFeature Set: {feature_name}"
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
        # Models
        # =========================

        for model_name, model in models.items():

            fold_scores = []

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

                acc = accuracy_score(
                    y_test,
                    y_pred
                )

                fold_scores.append(
                    acc
                )

            mean_acc = np.mean(
                fold_scores
            )

            std_acc = np.std(
                fold_scores
            )

            print(
                f"{model_name:<15}"
                f"{mean_acc:.4f}"
            )

            results.append([

                dataset_name,

                feature_name,

                model_name,

                mean_acc,

                std_acc
            ])
        # =========================
        # GMM
        # =========================

        gmm_scores = []

        for train_idx, test_idx in skf.split(X, y):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            # GMM需要Scale
            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(
                X_train
            )

            X_test_scaled = scaler.transform(
                X_test
            )

            y_pred = gmm_classifier(
                X_train_scaled,
                y_train,
                X_test_scaled,
                n_components=4
            )

            acc = accuracy_score(
                y_test,
                y_pred
            )

            gmm_scores.append(acc)

        gmm_mean = np.mean(gmm_scores)

        gmm_std = np.std(gmm_scores)

        print(
            f"{'GMM':<15}"
            f"{gmm_mean:.4f}"
        )

        results.append([
            dataset_name,
            feature_name,
            "GMM",
            gmm_mean,
            gmm_std
        ])
# =========================
# Save Result
# =========================

result_df = pd.DataFrame(

    results,

    columns=[

        "Preprocessing",

        "Feature_Set",

        "Model",

        "Mean_Accuracy",

        "Std"
    ]
)

result_df = result_df.sort_values(
    by="Mean_Accuracy",
    ascending=False
)

os.makedirs(
    "results",
    exist_ok=True
)

result_df.to_csv(
    "results/full_experiment.csv",
    index=False
)

print("\n")
print("=" * 60)
print("TOP 20 RESULTS")
print("=" * 60)

print(
    result_df.head(20)
)