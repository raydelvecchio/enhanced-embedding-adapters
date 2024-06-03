import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MiniLMWithAdapter(nn.Module):
    def __init__(self):
        super(MiniLMWithAdapter, self).__init__()
        self.minilm = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.dense_adapter = nn.Linear(384, 384)

        nn.init.normal_(self.dense_adapter.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.dense_adapter.bias)

    def forward(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors='pt')
        outputs = self.minilm(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0]  # representation of the [CLS] token
        dense_output = self.dense_adapter(pooled_output)
        return dense_output
