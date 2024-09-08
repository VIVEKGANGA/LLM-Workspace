import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, SequentialSampler, RandomSampler

def create_dataloader(txt: str, max_len: int = 256, stride: int = 128, batch_size: int = 4,
                      shuffle: bool = True, drop_last: bool = True, num_workers: int = 0) -> DataLoader:
    """
    Creates a PyTorch DataLoader from a given text.

    Args:
        txt (str): The input text to be tokenized and split into input and target sequences.
        max_len (int, optional): The maximum length of the input sequences. Defaults to 256.
        stride (int, optional): The stride for splitting the text into input and target sequences. Defaults to 128.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last batch if it's smaller than the batch size. Defaults to True.
        num_workers (int, optional): The number of worker processes for the DataLoader. Defaults to 0.

    Returns:
        DataLoader: A PyTorch DataLoader containing the input and target sequences.

    """
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(txt)
    input_ids = []
    target_ids = []

    for i in range(0, len(token_ids) - max_len, stride):
        target_ids.append(torch.tensor(token_ids[i+1:i+max_len+1]))
        input_ids.append(torch.tensor(token_ids[i:i+max_len]))

    input_ids = torch.cat(input_ids)
    target_ids = torch.cat(target_ids)

    dataset = TensorDataset(input_ids, target_ids)
    sampler = SequentialSampler(dataset) if not shuffle else RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=drop_last, num_workers=num_workers)
    return dataloader

        
with open('boy_knight.txt','r') as f:
    raw_text = f.read()
    
print("Total number of characters:", len(raw_text))
data_loader = create_dataloader(raw_text, batch_size=8, max_len=25, stride=25, shuffle=False)

data_iter = iter(data_loader)
inputs, target = next(data_iter)
print("Inputs:", inputs)
print("Target:", target)

