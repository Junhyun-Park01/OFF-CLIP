from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import numpy as np

def calculate_micro_macro_auprc(y_true, y_scores):

    n_labels = y_true.shape[1]
    macro_precisions = []
    macro_recalls = []
    for i in range(n_labels):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        macro_precisions.append(precision)
        macro_recalls.append(recall)
    macro_auprc = np.mean([auc(recall, precision) for precision, recall in zip(macro_precisions, macro_recalls)])

    # 微平均（Micro-average）
    y_true_combined = y_true.ravel()
    y_scores_combined = y_scores.ravel()
    precision, recall, _ = precision_recall_curve(y_true_combined, y_scores_combined)
    micro_auprc = auc(recall, precision)

    return micro_auprc, macro_auprc

def eval_auc(results, gt_labels):
    macro_auc = roc_auc_score(gt_labels, results , average="macro")
    micro_auc = roc_auc_score(gt_labels, results,  average="micro")
    weighted_auc = roc_auc_score(gt_labels, results,  average="weighted")
    per_auc = roc_auc_score(gt_labels, results,  average=None)
    return macro_auc, micro_auc, weighted_auc, per_auc

def compute_auc_threshold(gt, pred, n_class=5):
    gt_np = gt 
    pred_np = pred
    thresholds = []
    aucs = []
    select_best_thresholds = []
    for i in range(n_class):
        FP_rate_i, TP_rate_i, threshold_i = roc_curve(gt[:,i],  pred[:,i])
        J = TP_rate_i - FP_rate_i
        ix = np.argmax(J)
        best_threshold = threshold_i[ix]
        roc_auc_i = auc(FP_rate_i, TP_rate_i)
        thresholds.append(best_threshold)
        aucs.append(roc_auc_i)
    # print(f'auc: {aucs}\ntheshold: {thresholds}')
    return aucs, thresholds

def split_list(lst, chunk_size):
    result = []
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i+chunk_size]
        result.append(chunk)
    return result
