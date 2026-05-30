import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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

# model
model = SVC()

model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

# plot
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap="Blues")

plt.title("Genre Classification Confusion Matrix")
plt.show()