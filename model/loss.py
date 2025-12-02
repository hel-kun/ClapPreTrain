import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicts, targets):
        return self.mse_loss(predicts, targets)
    
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, predicts, targets):
        cosine_sim = self.cosine_similarity(predicts, targets)
        loss = 1 - cosine_sim.mean()
        return loss