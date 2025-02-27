import logging

from sklearn.metrics import confusion_matrix

# Configure logging for enterprise-level monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_misclassification_cost(y_true, y_pred, cost_fp=50, cost_fn=10):
    """
    Calculate the total cost of false positives (FP) and false negatives (FN).

    Args:
        y_true (array-like): Actual labels (0: Bad Quality, 1: Good Quality)
        y_pred (array-like): Predicted labels (0 or 1)
        cost_fp (float): Cost of a false positive (bad wine classified as good)
        cost_fn (float): Cost of a false negative (good wine classified as bad)

    Returns:
        float: Total cost of misclassifications.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = (fp * cost_fp) + (fn * cost_fn)

    logging.info(f"False Positives (FP): {fp}, False Negatives (FN): {fn}, Total Cost: ${total_cost}")

    return total_cost


def adjust_predictions(predictions, scale_factor=1.3):
    return predictions * scale_factor
