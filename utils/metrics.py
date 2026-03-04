from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, confusion_matrix

def compute_metrics(y_true, y_pred, threshold=0.4271):
    """
    Compute key performance metrics for sperm normality classification.
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate base metrics
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Adjusted Recall: recall^2 / (1 + FP/TP)
    fp_tp_ratio = fp / (tp + 1e-9)
    adj_recall = (recall ** 2.0) / (1 + fp_tp_ratio)
    
    # F2-Score: (5 * precision * recall) / (4 * precision + recall)
    if (4 * precision + recall) == 0:
        f2 = 0.0
    else:
        f2 = (5 * precision * recall) / (4 * precision + recall)
    
    metrics = {
        'ROC-AUC': roc_auc_score(y_true, y_pred),
        'PR-AUC (AP)': average_precision_score(y_true, y_pred),
        'F2-Score': f2,
        'Recall': recall,
        'Adjusted Recall': adj_recall
    }
    
    return metrics
