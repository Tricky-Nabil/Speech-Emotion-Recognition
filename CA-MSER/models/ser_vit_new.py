import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

class SER_ViT_NEW(nn.Module):
    """
    ViT model using timm with customization for SER task.
    
    Parameters
    ----------
    num_classes : int
        The number of output classes.
    in_ch   : int
        The number of input channels.
    img_size : int
        The input image size (square assumed, e.g., 224x224).
    pretrained : bool
        Load pretrained weights.
        
    Input
    -----
    Input dimension (N, C, H, W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width
    
    Output
    ------
    feature_map: Intermediate features (batch, hidden_dim)
    logits: Predicted class scores (batch, num_classes)
    """
    def __init__(self, num_classes=4, in_ch=3, img_size=224, pretrained=True):
        super(SER_ViT_NEW, self).__init__()
        # Initialize the ViT model
        self.vit_model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_ch,
            num_classes=num_classes
        )
        
        # The head layer in timm's ViT
        self.head = self.vit_model.get_classifier()
        self.vit_model.reset_classifier(0)  # Remove the head layer temporarily
        
        # Redefine the classifier layer for our task
        self.classifier = nn.Linear(self.vit_model.embed_dim, num_classes)
        init_layer(self.classifier)
        
        print('\n<< SER ViT Finetuning model initialized >>\n')

    def forward(self, x):
        
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Feature extraction using ViT encoder
        feature_map = self.vit_model.forward_features(x)
        print(feature_map.shape)
        # Classification output
        logits = self.classifier(feature_map)
        return feature_map, logits