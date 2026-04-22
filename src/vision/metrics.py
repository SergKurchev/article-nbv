import torch
import torch.nn.functional as F

def get_accuracy_difference(logits, target_class_id=None):
    """
    Computes Accuracy Difference measure.
    It is the difference between the probability of the best (Top-1) class
    and the alternative (Top-2) class.
    """
    probs = F.softmax(logits, dim=1)
    
    if probs.size(1) < 2:
        return torch.zeros(probs.size(0), device=probs.device)
        
    top2_probs, _ = torch.topk(probs, 2, dim=1)
    acc_diff = top2_probs[:, 0] - top2_probs[:, 1]
    
    return acc_diff
