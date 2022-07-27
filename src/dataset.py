import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Callable, List, Optional, Tuple


def pad_collate(batch, pad_token_idx):
    data_types = torch.tensor([ex[1] for ex in batch])
    batch_input_ids = [ex[0] for ex in batch]
    input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_idx)
    attention_mask = (input_ids != pad_token_idx).int()
    labels = pad_sequence(batch_input_ids, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels,
        "data_types": data_types
    }


class ArithmeticDataset(Dataset):

    def __init__(
        self, 
        numbers: List[Tuple[int]], 
        tokenizer,
        to_string_fn: Callable,
        data_type: int=0
    ):
        super().__init__()
        self.numbers = numbers
        self.tokenizer = tokenizer
        self.to_string_fn = to_string_fn
        self.data_type = torch.tensor(data_type)

    def __getitem__(self, index):
        string = self.to_string_fn(*self.numbers[index])
        encoded = self.tokenizer.encode(string)
        return torch.tensor(encoded), self.data_type

    def __len__(self):
        return len(self.numbers)

def pad_collate_with_hidden_reps(batch, pad_token_idx):
    data_types = torch.tensor([ex[1] for ex in batch])
    batch_input_ids = [ex[0] for ex in batch]
    # visual embeddings are of shape (num_chars_in_math_equation, embed_dim)
    # TODO: investigate if padding with 0s is best?
    batch_visual_embeddings = pad_sequence([ex[2] for ex in batch], batch_first=True, padding_value=0)
    visual_attention_mask = torch.all(batch_visual_embeddings != torch.zeros((batch_visual_embeddings.shape[2],)), dim=-1).int()
    input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=pad_token_idx)
    attention_mask = (input_ids != pad_token_idx).int()
    labels = pad_sequence(batch_input_ids, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids, 
        "visual_embeds": batch_visual_embeddings,
        "attention_mask": attention_mask, 
        "visual_attention_mask": visual_attention_mask,
        "labels": labels,
        "data_types": data_types
    }

class ArithmeticDatasetWithHiddenReps(Dataset):

    def __init__(
        self, 
        numbers: List[Tuple[int]], 
        tokenizer,
        to_string_fn: Callable,
        hidden_representations: dict,
        data_type: int=0
    ):
        super().__init__()
        self.numbers = numbers
        self.tokenizer = tokenizer
        self.to_string_fn = to_string_fn
        self.data_type = torch.tensor(data_type)
        self.hidden_representations = hidden_representations

    def __getitem__(self, index):
        string = self.to_string_fn(*self.numbers[index])
        encoded = self.tokenizer.encode(string)
        return torch.tensor(encoded), self.data_type, self.hidden_representations[self.numbers[index]]

    def __len__(self):
        return len(self.numbers)
