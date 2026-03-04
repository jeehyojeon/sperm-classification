import torch
import torch.nn as nn
import torch.nn.functional as F

class BitwiseValleyFocalLoss(nn.Module):
    """
    Bit-wise Valley Focal Loss implementation as used in the manuscript's 
    sensitivity studies, designed to handle label uncertainty and class imbalance.
    """
    def __init__(self, gamma=6.0, alpha=0.98, base_weights=None):
        super(BitwiseValleyFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        if base_weights is not None:
            self.register_buffer('weights', torch.tensor(base_weights, dtype=torch.float32))
        else:
            self.register_buffer('weights', torch.ones(6, dtype=torch.float32))
        
    def forward(self, logits, targets, strengths):
        # strengths should be the number of positive annotations (0-5)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(), targets, reduction='none'
        )
        probs = torch.sigmoid(logits.squeeze())
        # Correctly select p_t based on the target class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t for class balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss part
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * ce_loss
        
        # Bitwise Valley Weight part (Multiplicative based on annotation strength)
        w_s = self.weights[strengths]
            
        return (focal_loss * w_s).mean()

def get_loss_fn(config):
    """
    Helper to initialize the loss function from a configuration dictionary.
    """
    return BitwiseValleyFocalLoss(
        gamma=config.get('gamma', 6.0),
        alpha=config.get('alpha', 0.98),
        base_weights=config.get('bitwise_weights', [3.5, 1.0, 0.3, 0.3, 0.4, 0.5])
    )
