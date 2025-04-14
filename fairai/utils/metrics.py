import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


def disparate_impact(X, y, protected_attribute_name: str):
    feature = X[protected_attribute_name].to_numpy().flatten()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    if isinstance(feature, pd.Series):
        feature = feature.to_numpy()
    y = y.flatten()
    feature = feature.flatten()

    df = pd.DataFrame({"y": y, "feature": feature})
    numerator = df[df.feature == 0].y.mean()
    denominator = df[df.feature == 1].y.mean()

    di = numerator / denominator

    return di


def error_rate_difference(X, y_true, y_pred, protected_attribute_name):
    """Average odds difference is a performance metric that roughly tells us about
    the percentage of transactions incorrectly scored by the model.

    Args:
        X (_type_): _description_
        y_true (_type_): _description_
        y_pred (_type_): _description_
        protected_attribute_name (_type_): _description_
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()
    feature = X[protected_attribute_name].to_numpy().flatten()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "feature": feature})
    df_unpr = df[df.feature == 0].copy()
    df_pr = df[df.feature == 1].copy()

    # cm for unprivileged group
    tn, fp, fn, tp = confusion_matrix(df_unpr.y_true, df_unpr.y_pred).ravel()
    err_reference = (fp + fn) / (tn + fp + fn + tp)

    # cm for unprivileged group
    tn, fp, fn, tp = confusion_matrix(df_pr.y_true, df_pr.y_pred).ravel()
    err_monitored = (fp + fn) / (tn + fp + fn + tp)

    # average odds difference
    err_rate_diff = err_monitored - err_reference
    return err_rate_diff


def conditional_accuracy(X, y_true, y_pred, protected_attribute_name):
    if isinstance(y_true, pd.Series):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_numpy()

    feature = X[protected_attribute_name].to_numpy().flatten()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "feature": feature})
    df_unpr = df[df.feature == 0].copy()
    df_pr = df[df.feature == 1].copy()

    acc_unpr = round(accuracy_score(df_unpr.y_true, df_unpr.y_pred), 3)
    acc_pr = round(accuracy_score(df_pr.y_true, df_pr.y_pred), 3)

    return acc_unpr, acc_pr


def bias_fairness_report(X, y_true, y_pred_unmit, y_pred_mit, protected_attribute_name):
    bias_scores = {}
    bias_scores["accuracy"] = {
        "unmitigated": round(accuracy_score(y_true, y_pred_unmit), 3),
        "mitigated": round(accuracy_score(y_true, y_pred_mit), 3),
    }
    bias_scores["disparity"] = {
        "data": round(disparate_impact(X, y_true, protected_attribute_name), 3),
        "unmitigated": round(
            disparate_impact(X, y_pred_unmit, protected_attribute_name), 3
        ),
        "mitigated": round(
            disparate_impact(X, y_pred_mit, protected_attribute_name), 3
        ),
    }
    bias_scores["error_rate_difference"] = {
        "unmitigated": round(
            error_rate_difference(X, y_true, y_pred_unmit, protected_attribute_name), 3
        ),
        "mitigated": round(
            error_rate_difference(X, y_true, y_pred_mit, protected_attribute_name), 3
        ),
    }

    print(
        f"""
    Generating report for {protected_attribute_name} sensitive attribute...
    
    Disparity in Data: {bias_scores["disparity"]["data"]}
    
    Unmitigated Model
        Accuracy: {bias_scores["accuracy"]["unmitigated"]}
        Disparate Impact: {bias_scores["disparity"]["unmitigated"]}
        Error Rate Difference: {bias_scores["error_rate_difference"]["unmitigated"]}
    
    Mitigated Model
        Accuracy: {bias_scores["accuracy"]["mitigated"]}
        Disparate Impact: {bias_scores["disparity"]["mitigated"]}
        Error Rate Difference: {bias_scores["error_rate_difference"]["mitigated"]}"""
    )
    return bias_scores
