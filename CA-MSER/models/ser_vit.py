import torch
import torch.nn as nn
import torchvision
from torchvision.models.vision_transformer import vit_b_16

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

class SER_ViT(nn.Module):
    """
    Vision Transformer-based model for SER (Speech Emotion Recognition).
    
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    in_ch : int
        Number of input channels. Default is 3.
    pretrained : bool
        If True, initializes weights using a pre-trained ViT model.
    """

    def __init__(self, num_classes=4, in_ch=3, pretrained=True):
        super(SER_ViT, self).__init__()

        # Load the pre-trained ViT model
        self.vit_model = vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)

        # Modify the input channels if needed
        if in_ch != 3:
            self.vit_model.conv_proj = nn.Conv2d(in_ch, self.vit_model.conv_proj.out_channels, 
                                                 kernel_size=16, stride=16)
            init_layer(self.vit_model.conv_proj)

        # Replace the classification head for the required number of classes
        self.vit_model.heads.head = nn.Linear(self.vit_model.heads.head.in_features, num_classes)
        init_layer(self.vit_model.heads.head)

        print('\n<< SER ViT Finetuning model initialized >>\n')

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W).
        
        Returns
        -------
        feature_map : torch.Tensor
            Intermediate feature map from the ViT encoder.
        logits : torch.Tensor
            Output logits before applying Softmax.
        """
        # Feature map from the encoder
        feature_map = self.vit_model.encoder(self.vit_model._process_input(x))
        
        # Output logits
        logits = self.vit_model(x)

        return feature_map, logits
