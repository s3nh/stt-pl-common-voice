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

class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim : int, hidden_size : int, dropout : int, batch_first : bool) -> None:
        super(BidirectionalGRU, self).__init__()
        self.BiGRU = nn.GRU(input_size = rnn_dim, hidden_size = hidden_size, num_layers : int = 1, batch_first = batch_first, bidirectional: bool = True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        # flatten parameters, which is needed while training od dataparallell mode
        self.BiGRU.flatten_parameters()
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers : in, n_rnn_layers, : int, rnn_dim : int, n_class : int, n_feats : int, stride : int, dropout : float):
        super(SpeechRecognitionModel, self).__init__()
        self.in_channels: int = 1
        self.out_channels: int = 32
        self.kernel: int = 3
        self.padding: int = 3 // 2
        self.n_feats: int = n_feats // 2
        self.stride : int = 2
        self.dropout : float = dropout
        self.rnn_dim: int = rnn_dim
        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, self.stride, self.padding)

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel = self.kernel, stride = 1, dropout = self.dropout, n_feats = self.n_feats) for _ in range(n_cnn_layers)
        ])

        self.birnn_layers =  nn.Sequential(
            *[BidirectionalGRU(
                rnn_dim = rnn_dim if i == 0 else rnn_dim * 2, 
                hidden_size = rnn_dim, 
                dropout = self.dropout,
                batch_first = i == 0)
                for i in range(n_rnn_layers)]
            )

        self.fully_connected =  nn.Linear(self.n_feats * 32, rnn_dim)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x