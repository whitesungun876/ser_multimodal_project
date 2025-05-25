# src/evaluation/metrics.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_metrics(y_true, y_pred):
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    # Precision, Recall, F1 
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"→ Accuracy : {acc:.4f}")
    print(f"→ Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print("→ Confusion Matrix:")
    print(cm)
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
