import torch
import torch.nn as nn

class CRF(nn.Module):
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
    def forward(self, emissions, tags):
        """
        CRF的前向计算
        :param emissions: 发射概率矩阵，形状为 (batch_size, sequence_length, num_tags)
        :param tags: 真实标签序列，形状为 (batch_size, sequence_length)
        :return: CRF的负对数似然损失
        """
        batch_size, sequence_length = tags.shape
        scores = self._compute_scores(emissions, tags)
        log_partition = self._compute_log_partition(emissions)
        log_likelihood = scores - log_partition
        return -log_likelihood.mean()
    
    def _compute_scores(self, emissions, tags):
        batch_size, sequence_length, _ = emissions.shape
        scores = self.start_transitions[tags[:, 0]]
        for t in range(1, sequence_length):
            transition_scores = self.transitions[tags[:, t-1], tags[:, t]]
            emission_scores = emissions[:, t, torch.arange(batch_size), tags[:, t]]
            scores += transition_scores + emission_scores
        scores += self.end_transitions[tags[:, -1]]
        return scores
    
    def _compute_log_partition(self, emissions):
        batch_size, sequence_length, _ = emissions.shape
        alphas = self.start_transitions + emissions[:, 0]
        for t in range(1, sequence_length):
            broadcast_alphas = alphas.unsqueeze(-1)
            broadcast_emissions = emissions[:, t].unsqueeze(1)
            transition_scores = self.transitions.unsqueeze(0)
            scores = broadcast_alphas + transition_scores + broadcast_emissions
            alphas = torch.logsumexp(scores, dim=1)
        log_partition = torch.logsumexp(alphas + self.end_transitions, dim=1)
        return log_partition

# 使用示例
num_tags = 3  # 标签的数量

# 创建CRF模型
crf = CRF(num_tags)

# 定义输入数据
batch_size = 2
sequence_length = 3
num_features = 4
emissions = torch.randn(batch_size, sequence_length, num_tags)

# 定义真实标签序列
tags = torch.tensor([[0, 1, 2], [2, 0, 1]])

# 进行前向计算
loss = crf(emissions, tags)

print("Loss:", loss.item())
