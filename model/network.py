import torch.nn as nn
import torch.nn.functional as F

class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        """
        Parameters
        -----------
        n_feats : int
            Number of input features
        """
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int, 
                    dropout: int, n_feats: int):
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding = kernel//2)            
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding = kernel//2)
        self.dropout1 = nn.Dropout(dropout) 
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats=n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats = n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x