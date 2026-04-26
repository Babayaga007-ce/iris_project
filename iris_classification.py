import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── STEP 1: Load data ──────────────────────────────
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print(df.head())
print(df.describe())

# ── STEP 2: Visualise ──────────────────────────────
sns.pairplot(df, hue="species", palette="Set2")
plt.suptitle("Iris Pairplot", y=1.02)
plt.tight_layout()
plt.savefig("plot_1_pairplot.png", dpi=120, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feature in zip(axes.flatten(), iris.feature_names):
    for species, color in zip(iris.target_names, ["#4CAF50", "#2196F3", "#FF9800"]):
        ax.hist(df[df["species"] == species][feature], alpha=0.6,
                label=species, color=color, bins=15, edgecolor="white")
    ax.set_title(feature)
    ax.legend(fontsize=8)
plt.suptitle("Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("plot_2_histograms.png", dpi=120)
plt.show()

# ── STEP 3: Split ──────────────────────────────────
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# ── STEP 4: Scale ──────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── STEP 5: Train ──────────────────────────────────
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sc, y_train)

lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_sc, y_train)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)   # trees don't need scaling

# ── STEP 6: Evaluate ───────────────────────────────
models = {
    "KNN":                 (knn, X_test_sc),
    "Logistic Regression": (lr,  X_test_sc),
    "Decision Tree":       (dt,  X_test),
}

for name, (model, X_eval) in models.items():
    y_pred = model.predict(X_eval)
    print(f"\n── {name} ──")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ── STEP 7: Confusion matrices ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, (model, X_eval)) in zip(axes, models.items()):
    y_pred = model.predict(X_eval)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=iris.target_names).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("plot_3_confusion_matrices.png", dpi=120)
plt.show()

print("\nDone! PNG plots saved in your folder.")