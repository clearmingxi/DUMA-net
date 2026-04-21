import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_A, feature_B, logit_scale):
        # 1. 使用 softplus 保证温度始终平滑正数
        temp = F.softplus(logit_scale)

        # 2. 相似度矩阵（特征应事先 L2 归一化）
        logits = torch.matmul(feature_A, feature_B.t()) * temp
        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)

        # 3. 对称交叉熵：A→B 与 B→A 两个方向的 CE 之和
        loss_A2B = F.cross_entropy(logits, labels)
        loss_B2A = F.cross_entropy(logits.t(), labels)
        sce_loss = 0.5 * (loss_A2B + loss_B2A)

        return sce_loss
