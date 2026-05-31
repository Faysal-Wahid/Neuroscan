"""
=============================================================================
  DEMENTIA PREDICTION PROJECT
  Dataset: OASIS Longitudinal MRI Dataset (dementia_data.csv)
  Target:  Group (Nondemented / Demented / Converted)
  Task:    Multi-class Classification
=============================================================================

PROJECT OVERVIEW
----------------
This project predicts dementia status using longitudinal MRI brain scan data.
Three classes exist:
  - Nondemented : No cognitive impairment
  - Demented    : Diagnosed with dementia
  - Converted   : Was nondemented but converted to demented in a later visit

MODELS USED
-----------
Classical ML : Random Forest, SVM, Decision Tree, Gradient Boosting,
               Logistic Regression, KNN, XGBoost (if installed)
Deep Learning: LSTM, GRU (sequence models using visit-level longitudinal data)

SLR SUMMARY (Systematic Literature Review)
-------------------------------------------
1. METHODOLOGY
   - Papers reviewed from IEEE Xplore, PubMed, Google Scholar (2015–2024)
   - Keywords: "dementia prediction machine learning", "Alzheimer MRI classification",
     "OASIS dataset deep learning", "longitudinal dementia forecasting"
   - Inclusion criteria: peer-reviewed, used OASIS or similar MRI datasets,
     reported standard classification metrics
   - Common approaches found: SVM, Random Forest, CNN on MRI slices, LSTM for
     longitudinal data, ensemble methods, transformer-based architectures

2. GAPS / LIMITATIONS IN EXISTING LITERATURE
   - Most studies treat dementia as binary (demented vs. not), ignoring the
     "Converted" class which is clinically important for early intervention
   - Many studies use cross-sectional data; longitudinal modelling is underexplored
   - Class imbalance (Converted = only 37 samples here) is often not addressed
   - Feature importance and model explainability (e.g. SHAP) are rarely reported
   - Deep learning models lack interpretability, limiting clinical adoption
   - Small dataset sizes (OASIS has ~373 records) limit generalisability

3. METRICS REPORTED
   - Accuracy, Precision, Recall, F1-Score (weighted and macro)
   - Confusion Matrix, ROC-AUC (One-vs-Rest for multi-class)
   - Common benchmarks in literature: RF ~85–92%, SVM ~82–89%, LSTM ~83–90%

4. Run this code first then run streamlit
    -python -m pip install streamlit scikit-learn pandas numpy matplotlib seaborn
    -python -m streamlit run app.py
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Deep learning – PyTorch (comment out the DL section if not installed)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    DL_AVAILABLE = True
    print("✓ PyTorch found – LSTM and GRU will run.")
except ImportError:
    DL_AVAILABLE = False
    print("⚠ PyTorch not installed – deep learning section will be skipped.")
    print("  Install with:  pip install torch")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 1: LOADING DATA")
print("="*70)

df = pd.read_csv("dementia_data.csv")
print(f"Shape  : {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())
print("\nTarget distribution (Group):")
print(df["Group"].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 2: EDA")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Dementia Dataset – EDA", fontsize=15, fontweight="bold")

# 2a. Target distribution
ax = axes[0, 0]
colors = ["#4CAF50", "#F44336", "#FF9800"]
df["Group"].value_counts().plot(kind="bar", ax=ax, color=colors, edgecolor="black")
ax.set_title("Target Class Distribution")
ax.set_xlabel("Group"); ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width()/2, p.get_height() + 1),
                ha="center", fontsize=10)

# 2b. Age distribution by group
ax = axes[0, 1]
for g, c in zip(["Nondemented", "Demented", "Converted"], colors):
    subset = df[df["Group"] == g]["Age"]
    ax.hist(subset, bins=15, alpha=0.6, label=g, color=c, edgecolor="black")
ax.set_title("Age Distribution by Group")
ax.set_xlabel("Age"); ax.set_ylabel("Frequency")
ax.legend()

# 2c. MMSE by group (boxplot)
ax = axes[0, 2]
data_box = [df[df["Group"] == g]["MMSE"].dropna().values
            for g in ["Nondemented", "Demented", "Converted"]]
bp = ax.boxplot(data_box, patch_artist=True,
                labels=["Nondemented", "Demented", "Converted"])
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_title("MMSE Score by Group")
ax.set_ylabel("MMSE")

# 2d. CDR distribution
ax = axes[1, 0]
df["CDR"].value_counts().sort_index().plot(kind="bar", ax=ax,
                                            color="#5C6BC0", edgecolor="black")
ax.set_title("CDR Score Distribution")
ax.set_xlabel("CDR Score"); ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)

# 2e. nWBV vs Age scatter
ax = axes[1, 1]
for g, c in zip(["Nondemented", "Demented", "Converted"], colors):
    sub = df[df["Group"] == g]
    ax.scatter(sub["Age"], sub["nWBV"], label=g, alpha=0.5, color=c, s=30)
ax.set_title("Normalised Whole Brain Volume vs Age")
ax.set_xlabel("Age"); ax.set_ylabel("nWBV")
ax.legend()

# 2f. Correlation heatmap
ax = axes[1, 2]
num_cols = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ EDA plot saved → eda_plots.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 3: PREPROCESSING")
print("="*70)

# 3a. Encode Gender
df["M/F"] = LabelEncoder().fit_transform(df["M/F"])   # M=1, F=0

# 3b. Encode Target
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Group"])            # Demented=0, Nondemented=1, Converted=2 (sorted)
class_names = le.classes_
print(f"Class encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3c. Features
feature_cols = ["Visit", "MR Delay", "M/F", "Age", "EDUC", "SES",
                "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
X = df[feature_cols].values
y = df["Label"].values

# 3d. Impute + Scale
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3e. Train / Test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Features         : {X_train.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER – EVALUATE A MODEL
# ─────────────────────────────────────────────────────────────────────────────
results_summary = {}   # store all results for final comparison

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, class_names):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (One-vs-Rest)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)
        try:
            auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan
    else:
        # SVM without proba
        auc = np.nan

    results_summary[name] = dict(Accuracy=acc, Precision=prec,
                                  Recall=rec, F1=f1, AUC=auc)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}" if not np.isnan(auc) else "  ROC-AUC  : N/A")
    print(f"\nClassification Report:\n"
          f"{classification_report(y_te, y_pred, target_names=class_names, zero_division=0)}")

    return model, y_pred

# ─────────────────────────────────────────────────────────────────────────────
# 5. CLASSICAL ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 4: CLASSICAL MACHINE LEARNING MODELS")
print("="*70)

classical_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, class_weight="balanced"),
    "SVM (RBF Kernel)": SVC(
        kernel="rbf", C=10, gamma="scale", probability=False,
        class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
}

trained_models = {}
predictions    = {}

for name, model in classical_models.items():
    m, y_pred = evaluate_model(
        name, model, X_train, y_train, X_test, y_test, class_names)
    trained_models[name] = m
    predictions[name]    = y_pred

# ─────────────────────────────────────────────────────────────────────────────
# 6. DEEP LEARNING – LSTM & GRU  (requires PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 5: DEEP LEARNING MODELS (LSTM & GRU)")
print("="*70)

if DL_AVAILABLE:
    # For sequence models we group records by subject and treat each visit
    # as a timestep.  Because every row already contains "Visit" we sort
    # and pad to max_seq_len timesteps.

    # ── 6a. Build per-subject sequences ──────────────────────────────────
    df_raw = pd.read_csv("dementia_data.csv")
    df_raw["M/F"] = LabelEncoder().fit_transform(df_raw["M/F"])
    df_raw["Label"] = le.transform(df_raw["Group"])

    # impute numeric
    for col in ["SES", "MMSE"]:
        df_raw[col].fillna(df_raw[col].median(), inplace=True)

    # Assign a pseudo subject ID: rows with the same (Age-at-visit-1, M/F) group
    # (OASIS doesn't provide subject IDs in the exported CSV, so we use
    # the "MR Delay == 0" rows to anchor each subject.)
    feat = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF", "MR Delay"]
    df_raw = df_raw.sort_values(["Age", "M/F", "Visit"]).reset_index(drop=True)

    # Simple grouping heuristic: same consecutive (M/F, EDUC) block = same subject
    df_raw["SubjectID"] = (
        (df_raw[["M/F", "EDUC", "eTIV"]].shift() != df_raw[["M/F", "EDUC", "eTIV"]])
        .any(axis=1)
        .cumsum()
    )

    max_seq = 5          # maximum number of visits per subject
    n_features = len(feat)

    sequences, labels_seq = [], []
    sc2 = StandardScaler()
    df_raw[feat] = sc2.fit_transform(df_raw[feat])

    for sid, grp in df_raw.groupby("SubjectID"):
        grp = grp.sort_values("Visit")
        seq = grp[feat].values[:max_seq]
        # zero-pad if fewer than max_seq visits
        pad = np.zeros((max_seq - len(seq), n_features))
        seq = np.vstack([seq, pad])
        sequences.append(seq)
        labels_seq.append(grp["Label"].iloc[-1])   # label = last known status

    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(labels_seq, dtype=np.int64)

    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

    def make_loader(X, y, batch=16, shuffle=True):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    train_loader = make_loader(X_tr_s, y_tr_s)
    test_loader  = make_loader(X_te_s, y_te_s, shuffle=False)

    n_classes = len(np.unique(y_seq))

    # ── 6b. Model definitions ─────────────────────────────────────────────
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc   = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])   # last timestep

    class GRUClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=dropout)
            self.fc  = nn.Linear(hidden_size, num_classes)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    def train_dl_model(model_name, model, epochs=60, lr=1e-3):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        train_losses, val_accs = [], []

        for epoch in range(epochs):
            model.train()
            ep_loss = 0
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                out  = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
            scheduler.step()
            train_losses.append(ep_loss / len(train_loader))

            # validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    preds = model(X_b).argmax(1)
                    correct += (preds == y_b).sum().item()
                    total   += len(y_b)
            val_accs.append(correct / total)

            if (epoch + 1) % 10 == 0:
                print(f"  [{model_name}] Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {train_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}")

        # Final evaluation
        model.eval()
        all_preds, all_true, all_probs = [], [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                logits = model(X_b)
                probs  = torch.softmax(logits, dim=1)
                preds  = logits.argmax(1)
                all_preds.extend(preds.numpy())
                all_true.extend(y_b.numpy())
                all_probs.extend(probs.numpy())

        y_pred_dl  = np.array(all_preds)
        y_true_dl  = np.array(all_true)
        y_probs_dl = np.array(all_probs)

        acc  = accuracy_score(y_true_dl, y_pred_dl)
        prec = precision_score(y_true_dl, y_pred_dl, average="weighted", zero_division=0)
        rec  = recall_score(y_true_dl, y_pred_dl, average="weighted", zero_division=0)
        f1   = f1_score(y_true_dl, y_pred_dl, average="weighted", zero_division=0)
        try:
            auc = roc_auc_score(y_true_dl, y_probs_dl, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan

        results_summary[model_name] = dict(
            Accuracy=acc, Precision=prec, Recall=rec, F1=f1, AUC=auc)
        predictions[model_name] = y_pred_dl

        print(f"\n{'─'*50}")
        print(f"  {model_name} – Final Results")
        print(f"{'─'*50}")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}" if not np.isnan(auc) else "  ROC-AUC  : N/A")
        print(f"\nClassification Report:\n"
              f"{classification_report(y_true_dl, y_pred_dl, target_names=class_names, zero_division=0)}")

        return train_losses, val_accs, y_true_dl, y_pred_dl

    # ── 6c. Train LSTM ────────────────────────────────────────────────────
    print("\n--- Training LSTM ---")
    lstm_model = LSTMClassifier(n_features, hidden_size=64,
                                num_layers=2, num_classes=n_classes)
    lstm_losses, lstm_accs, y_true_lstm, y_pred_lstm = train_dl_model(
        "LSTM", lstm_model, epochs=60)

    # ── 6d. Train GRU ─────────────────────────────────────────────────────
    print("\n--- Training GRU ---")
    gru_model = GRUClassifier(n_features, hidden_size=64,
                              num_layers=2, num_classes=n_classes)
    gru_losses, gru_accs, y_true_gru, y_pred_gru = train_dl_model(
        "GRU", gru_model, epochs=60)

    # ── 6e. Training curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Deep Learning Training Curves", fontweight="bold")
    for ax, losses, accs, name in zip(axes,
                                       [lstm_losses, gru_losses],
                                       [lstm_accs,   gru_accs],
                                       ["LSTM", "GRU"]):
        ax2 = ax.twinx()
        ax.plot(losses, label="Train Loss", color="#E53935")
        ax2.plot(accs,  label="Val Accuracy", color="#1E88E5", linestyle="--")
        ax.set_title(f"{name} Training Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss", color="#E53935")
        ax2.set_ylabel("Accuracy", color="#1E88E5")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("dl_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ DL training curves saved → dl_training_curves.png")

else:
    print("  Skipping LSTM/GRU – PyTorch not available.")
    print("  Install PyTorch (https://pytorch.org) and re-run.")

# ─────────────────────────────────────────────────────────────────────────────
# 7. CONFUSION MATRICES – ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 6: CONFUSION MATRICES")
print("="*70)

n_models  = len(predictions)
n_cols    = 4
n_rows    = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * 4.5, n_rows * 4))
axes = axes.flatten()

for idx, (name, y_pred_) in enumerate(predictions.items()):
    y_true_cm = y_test if name not in ["LSTM", "GRU"] else (
        y_true_lstm if name == "LSTM" else y_true_gru)
    cm = confusion_matrix(y_true_cm, y_pred_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[idx], linewidths=0.5)
    axes[idx].set_title(name, fontsize=10, fontweight="bold")
    axes[idx].set_xlabel("Predicted"); axes[idx].set_ylabel("Actual")

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Confusion Matrices – All Models", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Confusion matrices saved → confusion_matrices.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL COMPARISON TABLE & PLOT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 7: MODEL COMPARISON")
print("="*70)

results_df = pd.DataFrame(results_summary).T.round(4)
results_df.sort_values("F1", ascending=False, inplace=True)
print(results_df.to_string())

fig, ax = plt.subplots(figsize=(14, 6))
metrics   = ["Accuracy", "Precision", "Recall", "F1"]
n         = len(results_df)
x         = np.arange(n)
width     = 0.18
bar_colors = ["#1E88E5", "#43A047", "#FB8C00", "#E53935"]

for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
    bars = ax.bar(x + i * width, results_df[metric].values,
                  width, label=metric, color=color, alpha=0.85, edgecolor="black")

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df.index, rotation=30, ha="right", fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Comparison – Accuracy / Precision / Recall / F1",
             fontweight="bold")
ax.legend(loc="upper right")
ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="0.90 threshold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Model comparison plot saved → model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE IMPORTANCE (Random Forest + Gradient Boosting)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 8: FEATURE IMPORTANCE")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, mname in zip(axes, ["Random Forest", "Gradient Boosting"]):
    importances = trained_models[mname].feature_importances_
    indices     = np.argsort(importances)[::-1]
    ax.bar(range(len(feature_cols)),
           importances[indices], color="#5C6BC0", edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in indices],
                       rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Feature Importance – {mname}", fontweight="bold")
    ax.set_ylabel("Importance")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Feature importance plot saved → feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. CROSS-VALIDATION (5-fold)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  STEP 9: 5-FOLD CROSS VALIDATION")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in classical_models.items():
    scores = cross_val_score(model, X_scaled, y,
                             cv=cv, scoring="f1_weighted", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:30s}  CV F1 = {scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
names_cv = list(cv_results.keys())
means    = [cv_results[n].mean() for n in names_cv]
stds     = [cv_results[n].std()  for n in names_cv]

bars = ax.bar(names_cv, means, yerr=stds, capsize=5,
              color="#26A69A", edgecolor="black", alpha=0.85)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Weighted F1 Score")
ax.set_title("5-Fold Cross-Validation – Weighted F1 Score", fontweight="bold")
ax.tick_params(axis="x", rotation=30)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, m + 0.02,
            f"{m:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("crossval_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Cross-validation plot saved → crossval_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# 11. FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  FINAL RESULTS SUMMARY")
print("="*70)
print(results_df.to_string())
best = results_df["F1"].idxmax()
print(f"\n  ★ Best model by F1: {best}  (F1 = {results_df.loc[best, 'F1']:.4f})")

print("""
─────────────────────────────────────────────────────────────────────────────
  FILES GENERATED
─────────────────────────────────────────────────────────────────────────────
  eda_plots.png            – Exploratory data analysis
  confusion_matrices.png   – Confusion matrix for every model
  model_comparison.png     – Side-by-side metric comparison bar chart
  feature_importance.png   – RF & GB feature importances
  crossval_results.png     – 5-fold CV F1 scores
  dl_training_curves.png   – LSTM & GRU training curves (if PyTorch installed)
─────────────────────────────────────────────────────────────────────────────
""")
