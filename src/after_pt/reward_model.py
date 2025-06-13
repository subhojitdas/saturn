import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        device = "mps"
        self.base_model = base_model.to(device)
        self.tokenizer = tokenizer.to(device)
        # scalar reward head
        self.v_head = nn.Linear(self.base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        reward = self.v_head(last_hidden_state).squeeze(-1)
        #its a trick to get around with padded tokens [1,1,1,1,1,0,0,0,0]
        final_reward = reward.gather(1, attention_mask.sum(dim=1).unsqueeze(1) - 1).squeeze(1)
        return final_reward