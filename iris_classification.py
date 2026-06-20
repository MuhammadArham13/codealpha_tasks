import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv("data/Iris.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())


print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== CLASS DISTRIBUTION ==========")
print(df["Species"].value_counts())


sns.set_style("whitegrid")

pair = sns.pairplot(df, hue="Species")
pair.savefig("screenshots/pairplot.png")
plt.close()


df.hist(figsize=(10,8))
plt.tight_layout()
plt.savefig("screenshots/histograms.png")
plt.close()



features = [
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]

for col in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"screenshots/{col}_boxplot.png")
    plt.close()


plt.figure(figsize=(8,5))
sns.countplot(x='Species', data=df)
plt.title("Species Distribution")
plt.savefig("screenshots/countplot.png")
plt.close()


temp_df = df.copy()

temp_df["Species"] = LabelEncoder().fit_transform(
    temp_df["Species"]
)

plt.figure(figsize=(8,6))
sns.heatmap(
    temp_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("screenshots/heatmap.png")
plt.close()


df = df.drop("Id", axis=1)

encoder = LabelEncoder()
df["Species"] = encoder.fit_transform(df["Species"])

X = df.drop("Species", axis=1)
y = df["Species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = []

best_accuracy = 0
best_model = None
best_name = ""

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    print("\n================================")
    print(name)
    print("================================")

    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\n========== MODEL COMPARISON ==========")
print(results_df)


predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title(
    f"Confusion Matrix ({best_name})"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "screenshots/confusion_matrix.png"
)

plt.close()


with open(
    "models/best_model.pkl",
    "wb"
) as file:
    pickle.dump(best_model, file)

print("\nBest Model:", best_name)
print("Accuracy:", best_accuracy)



print("\n====== CUSTOM PREDICTION ======")

sepal_length = float(input("Sepal Length: "))
sepal_width = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width = float(input("Petal Width: "))

sample = np.array([
    [
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]
])

prediction = best_model.predict(sample)

species = encoder.inverse_transform(prediction)

print(
    "\nPredicted Species:",
    species[0]
)