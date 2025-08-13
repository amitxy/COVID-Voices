import numpy as np
import evaluate


_MACRO_METRIC_NAMES = ["f1", "precision", "recall"]
_BASE_METRIC_NAMES  = ["accuracy"]
_ROC_METRIC_NAMES = ["roc_auc_mc"]

_METRICS = { name:evaluate.load(name) for name in _MACRO_METRIC_NAMES + _BASE_METRIC_NAMES }
_METRICS['roc_auc_mc'] = evaluate.load("roc_auc", "multiclass")

def _softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax -> probabilities (float32)
    """
    x = np.asarray(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return (ex / np.sum(ex, axis=axis, keepdims=True)).astype(np.float32)

def compute_metrics(eval_pred: tuple) -> dict:
    """Compute accuracy, macro-F1, macro-precision, macro-recall, and ROC AUC"""
    logits, labels = eval_pred
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    probabilities = _softmax_stable(logits)
    
    # Argmax predictions
    predictions = np.argmax(probabilities, axis=-1)

    # Macro metrics
    results = {name: _METRICS[name].compute(predictions=predictions, references=labels, average="macro")[name]
               for name in _MACRO_METRIC_NAMES}

    results.update(_METRICS["accuracy"].compute(predictions=predictions, references=labels))
    results.update(_METRICS["roc_auc_mc"].compute(prediction_scores=probabilities, references=labels, average="macro", multi_class="ovr" ))

    return results