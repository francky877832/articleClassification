#plot from local
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib

# === Load Data ===
labels = np.load('./results/labels.npy')
logits = np.load('./results/predictions.npy')

# Apply softmax manually
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # To prevent overflow
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Normalize

preds = np.argmax(probs, axis=1)

label_encoder = joblib.load('./results/label_encoder.pkl')
class_names = label_encoder.classes_
num_classes = len(class_names)

# === Metrics ===
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average='macro')
recall = recall_score(labels, preds, average='macro')
f1 = f1_score(labels, preds, average='macro')

# Sensitivity = Recall, Specificity needs custom implementation
cm = confusion_matrix(labels, preds)
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (TP + FP + FN)
sensitivity = np.mean(TP / (TP + FN))  # recall
specificity = np.mean(TN / (TN + FP))

# AUC
y_bin = label_binarize(labels, classes=range(num_classes))
auc_score = roc_auc_score(y_bin, probs, multi_class='ovr')

# Save all metrics
metrics_dict = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'F1-Score': f1,
    'AUC': auc_score
}
os.makedirs('./results', exist_ok=True)
with open('./results/final_metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("\n=== Evaluation Metrics ===")
with open('./results/printed_final_metrics.txt', 'w') as f_out:
    for k, v in metrics_dict.items():
        line = f"{k}: {v:.4f}"
        print(line)
        f_out.write(line + '\n')

# === Confusion Matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('./results/confusion_matrix.png')
plt.show()

# === ROC Curve ===
fpr = dict()
tpr = dict()
roc_auc = dict()
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('./results/roc_curve.png')
plt.show()

# === Loss Plot ===
log_history = np.load('./results/log_history.npy', allow_pickle=True)
train_steps, train_loss, eval_steps, eval_loss = [], [], [], []
for log in log_history:
    if 'loss' in log:
        train_steps.append(log['step'])
        train_loss.append(log['loss'])
    if 'eval_loss' in log:
        eval_steps.append(log['step'])
        eval_loss.append(log['eval_loss'])

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label='Train Loss')
plt.plot(eval_steps, eval_loss, label='Eval Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Train vs Eval Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./results/loss_curve.png')
plt.show()

# === Training/Inference Times ===
with open('./results/times.txt', 'r') as f:
    times = f.read()

print("\n=== Training and Inference Times ===")
print(times)
with open('./results/printed_times.txt', 'w') as f_out:
    f_out.write(times)


