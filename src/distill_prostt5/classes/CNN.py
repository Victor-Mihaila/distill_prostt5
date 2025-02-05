import torch.nn as nn
import torch

"""
Taken From ProstT5/Phold
"""

# Convolutional neural network (two convolutional layers)
class CNN(nn.Module):
    def __init__(self):
        """
        Initialize the Convolutional Neural Network (CNN) model.
        """
        super(CNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, num_classes).

        L = protein length
        B = batch-size
        F = number of features (1024 for embeddings)
        N = number of classes (20 for 3Di)
        """

        # Permute input tensor to match expected shape
        # Input shape: (batch_size, sequence_length, embedding_size)
        # Output shape: (batch_size, embedding_size, sequence_length, 1)

        x = x.permute(0, 2, 1).unsqueeze(
            dim=-1
        )  # IN: X = (B x L x F); OUT: (B x F x L, 1)

        # Pass the input through the classifier
        # Output shape: (batch_size, num_classes, sequence_length, 1)
        Yhat = self.classifier(x)  # OUT: Yhat_consurf = (B x N x L x 1)

        # Remove the singleton dimension from the output tensor
        # Output shape: (batch_size, num_classes, sequence_length)
        Yhat = Yhat.squeeze(dim=-1)  # IN: (B x N x L x 1); OUT: ( B x L x N )
        return Yhat