from torch import nn
from torch.types import Tensor

from transformers import BertModel

class DesinfoVacinalModel(nn.Module):
    """BERT-based model for Desinfo Vacinal Project.

    Args:
        nn.Module: Base class for all neural network modules in PyTorch.
    """
    def __init__(self, pretrained_model_name, num_labels):
        super().__init__()
        
        # Loads a pre-trained BERT model
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.3)
        
        # Classifier layer to output logits for each class
        self.classifier = nn.Linear(
            in_features=self.bert.config.hidden_size, 
            out_features=num_labels
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs.
            attention_mask (torch.Tensor): Tensor indicating which tokens to attend to.

        Returns:
            logits (torch.Tensor): Logits for each class.
        """
        # Fowards the inputs through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extracts the pooled output from BERT's outputs
        pooled_output = outputs.pooler_output
        
        # Applies dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Passes the pooled output through the classifier to get logits
        logits = self.classifier(pooled_output)
        
        return logits