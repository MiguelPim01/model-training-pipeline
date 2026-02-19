import torch
from torch.utils.data import Dataset


class DesinfoVacinalDataset(Dataset):
    """Dataset class for Desinfo Vacinal Project.

    Args:
        Dataset (_type_): Interface for PyTorch datasets.
    """

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Allows Pytorch to infer the size of the dataset.

        Returns:
            int (_type_): Size of the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """Returns a specific item from the dataset.

        Args:
            tensors (dict[str, torch.Tensor]): A dictionary with:
                - "input_ids": Token IDs tensor of shape (max_length,)
                - "attention_mask": Attention mask tensor of shape (max_length,)
                - "label": Label tensor
        """
        # Gets the item at the specified index
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenizes the text using the provided tokenizer
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Returns tensors for input IDs, attention mask, and label
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }
