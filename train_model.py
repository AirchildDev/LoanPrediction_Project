import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

MODEL_DIR = "model"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv("data/train.csv")
df = df.drop(columns=["Loan_ID"])
df = df.dropna(subset=["Loan_Status"])

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"].map({"Y": 1, "N": 0})

numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
categorical_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Step 1: Compare five algorithms

candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("Comparing candidate algorithms using 5-fold cross-validation...\n")
for name, clf in candidates.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    acc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    roc_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    results.append({
        "Model": name,
        "CV Accuracy": acc_scores.mean(),
        "CV F1": f1_scores.mean(),
        "CV ROC-AUC": roc_scores.mean(),
    })
    print(f"{name:22s} | Accuracy: {acc_scores.mean():.4f} | F1: {f1_scores.mean():.4f} | ROC-AUC: {roc_scores.mean():.4f}")

results_df = pd.DataFrame(results).sort_values("CV ROC-AUC", ascending=False).reset_index(drop=True)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

best_model_name = results_df.loc[0, "Model"]
print(f"\nBest model selected: {best_model_name} (highest CV ROC-AUC)")


# Step 2: Train the best model on the full training split

best_classifier = candidates[best_model_name]
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", best_classifier)
])
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nFinal held-out test accuracy ({best_model_name}):", accuracy)
print(report)


# Save the trained model 

model_path = os.path.join(MODEL_DIR, "loan_model.pkl")
joblib.dump(model_pipeline, model_path)
print(f"Model saved to {model_path}")


# Save the metrics report 

report_path = os.path.join(OUTPUT_DIR, "metrics_report.txt")
with open(report_path, "w") as f:
    f.write(f"Best model: {best_model_name}\n\n")
    f.write("Model comparison (5-fold CV on training set):\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nFinal held-out test accuracy: {accuracy}\n\n")
    f.write(report)
print(f"Metrics saved to {report_path}")


# Visualization 1: Model Comparison 

plt.figure(figsize=(8, 5))
plot_df = results_df.melt(id_vars="Model", value_vars=["CV Accuracy", "CV F1", "CV ROC-AUC"],
                           var_name="Metric", value_name="Score")
sns.barplot(x="Score", y="Model", hue="Metric", data=plot_df)
plt.title("Algorithm Comparison (5-Fold Cross-Validation)")
plt.xlim(0, 1)
plt.tight_layout()
comp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
plt.savefig(comp_path, dpi=150)
plt.close()
print(f"Saved {comp_path}")


# Visualization 2: Confusion Matrix (best model) 

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap="Blues", ax=ax, colorbar=False)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved {cm_path}")


# Visualization 3: Feature Importance (only if supported) 

classifier_step = model_pipeline.named_steps["classifier"]
if hasattr(classifier_step, "feature_importances_"):
    encoder = model_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_names = encoder.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(encoded_cat_names)

    importances = classifier_step.feature_importances_
    importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True).tail(10)

    plt.figure(figsize=(8, 6))
    sns.barplot(x="importance", y="feature", data=importance_df, color="steelblue")
    plt.title(f"Top 10 Feature Importances - {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150)
    plt.close()
    print(f"Saved {fi_path}")
else:
    print(f"{best_model_name} does not expose feature_importances_ — skipping feature importance plot.")