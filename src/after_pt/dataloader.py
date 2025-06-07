from torch.utils.data import Dataset

class PreferenceDatasetLite(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        d = self.data[idx]
        chosen_input = self.tokenizer(d['prompt'] + d['chosen'],
                  return_tensors='pt',
                  padding='max_length',
                  max_length=self.max_length,
                  truncation=True)
        rejected_output = self.tokenizer(d['prompt'] + d['rejected'],
                  return_tensors='pt',
                  padding='max_length',
                  max_length=self.max_length,
                  truncation=True)
        return chosen_input, rejected_output

    def __len__(self):
        return len(self.data)