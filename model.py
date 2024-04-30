import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.25):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.dropout1 = Dropout(dropout_rate)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm1d(hidden_channels)
        self.dropout2 = Dropout(dropout_rate)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm1d(hidden_channels)
        self.dropout3 = Dropout(dropout_rate)

        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = BatchNorm1d(hidden_channels)
        self.dropout4 = Dropout(dropout_rate)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x).relu()
        x = self.dropout3(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x).relu()
        x = self.dropout4(x)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x).relu()
        x = self.lin2(x)

        return x