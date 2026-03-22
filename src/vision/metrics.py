import torch
import torch.nn.functional as F

def get_accuracy_difference(logits, target_class_id):
    """
    Computes Accuracy Difference measure.
    It is the difference between the probability of the target class
    and the highest probability among incorrect classes.
    """
    probs = F.softmax(logits, dim=1)
    
    # Target prob
    target_prob = probs[0, target_class_id].item()
    
    # Best err prob
    probs_clone = probs.clone()
    probs_clone[0, target_class_id] = -1.0
    best_err_prob = probs_clone.max(dim=1)[0].item()
    
    return target_prob - best_err_prob
