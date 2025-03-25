import numpy as np

def calculate_metrics(conf_mat:np.ndarray) -> list[float]:
    '''
    Calculates and prints Accuracy, Precision, Recall, F1
    '''
    tp_M = conf_mat[1][1]
    tn_M = conf_mat[0][0]
    fn_M = conf_mat[1][0]
    fp_M = conf_mat[0][1]

    Accuracy = (tp_M + tn_M) / (tp_M + tn_M + fp_M + fn_M)
    print("Accuracy = ", Accuracy)
    Precision = tp_M / (tp_M + fp_M)
    print("Precision = ", Precision)
    Recall = tp_M / (tp_M + fn_M)
    print("Recall = ", Recall)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    print("F1 = ", F1)

    return [Accuracy, Precision, Recall, F1]
