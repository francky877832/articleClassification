#plot from drive
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
from google.colab import drive

# === Mount Google Drive ===
drive.mount('/content/drive')

# Define the paths on Google Drive
drive_results_path = '/content/drive/My Drive/article_classification/gpt2/results'
drive_labels_path = os.path.join(drive_results_path, 'labels.npy')
drive_logits_path = os.path.join(drive_results_path, 'predictions.npy')
drive_label_encoder_path = os.path.join(drive_results_path, 'label_encoder.pkl')
drive_log_history_path = os.path.join(drive_results_path, 'log_history.npy')
drive_times_path = os.path.join(drive_results_path, 'times.txt')

# === Load Data ===
labels = np.load(drive_labels_path)
logits = np.load(drive_logits_path)

# Apply softmax manually
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # To prevent overflow
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Normalize

preds = np.argmax(probs, axis=1)

label_encoder = joblib.load(drive_label_encoder_path)
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
os.makedirs(drive_results_path, exist_ok=True)
with open(os.path.join(drive_results_path, 'final_metrics.json'), 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("\n=== Evaluation Metrics ===")
with open(os.path.join(drive_results_path, 'printed_final_metrics.txt'), 'w') as f_out:
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
plt.savefig(os.path.join(drive_results_path, 'confusion_matrix.png'))
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
plt.savefig(os.path.join(drive_results_path, 'roc_curve.png'))
plt.show()

# === Loss Plot ===
log_history = np.load(drive_log_history_path, allow_pickle=True)
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
plt.savefig(os.path.join(drive_results_path, 'loss_curve.png'))
plt.show()

# === Training/Inference Times ===
with open(drive_times_path, 'r') as f:
    times = f.read()

print("\n=== Training and Inference Times ===")
print(times)
with open(os.path.join(drive_results_path, 'printed_times.txt'), 'w') as f_out:
    f_out.write(times)
