import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve, accuracy_score, f1_score, recall_score

def count_accuracy(B_true, B_prob):
    """
    计算因果发现的常用指标。
    B_true: 真实邻接矩阵 (N, N) (binary)
    B_prob: 预测的权重矩阵 (N, N) (continuous)
    注意：默认忽略对角线元素。
    """
    # 展平并移除对角线
    n = B_true.shape[0]
    # 获取非对角线的 mask
    mask = ~np.eye(n, dtype=bool)
    
    true_flat = B_true[mask].flatten()
    prob_flat = B_prob[mask].flatten()
    
    # 1. AUROC
    if len(np.unique(true_flat)) < 2:
        # 如果 Ground Truth 全是 0 或全是 1，无法计算 AUC，返回 0.5 或 0
        auroc = 0.5
        auprc = 0.0
    else:
        fpr, tpr, _ = roc_curve(true_flat, prob_flat)
        auroc = auc(fpr, tpr)
        
        # 2. AUPRC (Average Precision)
        precision, recall, _ = precision_recall_curve(true_flat, prob_flat)
        auprc = auc(recall, precision)
    
    # 3. Binary Metrics (F1, ACC, Recall, Precision)
    best_f1 = 0.0
    best_acc = 0.0
    best_recall = 0.0
    best_thresh = 0.0
    
    # 采样 100 个阈值进行搜索
    if prob_flat.max() > prob_flat.min():
        thresholds = np.linspace(prob_flat.min(), prob_flat.max(), 100)
    else:
        thresholds = [prob_flat.min()]

    for th in thresholds:
        pred_binary = (prob_flat > th).astype(int)
        
        f1 = f1_score(true_flat, pred_binary, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            best_acc = accuracy_score(true_flat, pred_binary)
            best_recall = recall_score(true_flat, pred_binary, zero_division=0)
            
    return {
        'AUROC': float(auroc),
        'AUPRC': float(auprc),
        'F1': float(best_f1),
        'Recall': float(best_recall),
        'ACC': float(best_acc),
        'Best_Threshold': float(best_thresh)
    }