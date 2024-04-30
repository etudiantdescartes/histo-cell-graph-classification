from torch_geometric.loader import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from model import GCN
from preprocessing import GraphDataset

def train(loader):
    """
    Training over one epoch
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        data.edge_index = data.edge_index.to(torch.int64)
        out = model(data.x, data.edge_index, data.batch)#data.edge_weight,
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validation(loader):
    """
    Validation over one epoch
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.edge_index = data.edge_index.to(torch.int64)
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)


def plot_training_curves(train_curve, val_curve):
    plt.plot(train_curve, label='Train loss', color='blue')
    plt.plot(val_curve, label='Val loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.legend()
    plt.show()


def delete_metadata(dataset):
    """
    Deleting metadata from train val sets to reduce the size of the graph objects and speed up training
    """
    for data in tqdm(dataset):
        if 'metadata' in data:
            del data.metadata


def train_loop(train_loader, val_loader):
    train_curve = []
    val_curve = []
    min_val_loss = np.inf
    for epoch in range(num_epochs):
        train_loss = train(train_loader)
        val_loss = validation(val_loader)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        
        #Save model every time val_loss reaches a new min value
        if val_loss < min_val_loss:
            torch.save(model, 'model/gnn_best_model.pt')
            min_val_loss = val_loss
            print('Model saved')
    return train_curve, val_curve

    
def accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

    accuracy = correct / total

    return accuracy
    
    
if __name__ == '__main__':
    in_channels=6#length of the nodes feature vectors
    hidden_channels = 64
    num_classes=2
    model = GCN(in_channels, hidden_channels, num_classes)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 10
    
    train_dataset = torch.load('torch_datasets/train_set.pt')
    val_dataset = torch.load('torch_datasets/val_set.pt')
    
    delete_metadata(train_dataset)
    delete_metadata(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')
    
    model = model.to(device)
    t = time.time()
    train_curve, val_curve = train_loop(train_loader, val_loader)
    print(f'Training time: {time.time()-t} seconds')
    plot_training_curves(train_curve, val_curve)
    
    #Evaluation on test set
    test_dataset = torch.load('torch_datasets/test_set.pt')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    mod = torch.load('model/gnn_best_model.pt')#Load best model saved during training
    mod = mod.to(device)
    accuracy(test_loader, mod)
    
    
    