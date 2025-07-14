import logging
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
import pickle
import time

import Configs
import Utils
from GenerateData import WaterGraphDataset

class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads, dropout=0.5)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=0.5)
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.gru1 = torch.nn.GRU(
            input_size=32,
            hidden_size=hidden_channels,
            batch_first=True
        )
        self.gru2 = torch.nn.GRU(
            input_size=hidden_channels,
            hidden_size=hidden_channels,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        batch_size, num_nodes, num_timesteps = x.size()
        x = x.reshape(batch_size * num_nodes, num_timesteps)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = self.norm(x)
        x = x.view(batch_size, num_nodes, -1)
        x = x.permute(0, 2, 1) # [batch_size, num_timesteps, num_nodes]

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.fc(x)
        return x

def Tester(args, model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    acc_sums = {k: 0.0 for k in args.topk}
    acc_counts = 0
    losses = []
    with torch.no_grad():
        for data in loader:
            x = data.x.to(device) # [batch_size*number_of_nodes, number_of_timesteps] [2048, 48]
            edge_index = data.edge_index.to(device) # [2, batch_size*number_of_edges] [2, 2176]
            y = data.y.to(device) # [batch_size] [64]
            x = x.view(len(y), -1, x.size(-1)) # [batch_size, number_of_nodes, number_of_timesteps] [64, 32, 48]
            out = model(x, edge_index) # [batch_size, number_of_nodes + 1]
            loss = criterion(out, y)
            acc_dict, topk_preds, topk_probs = Utils.accuracy(out, y, topk=args.topk)
            for k in args.topk:
                acc_sums[k] += acc_dict[k]

            acc_counts += 1
            losses.append(loss.item())

    # Compute average top-k accuracy
    avg_accs = {k: acc_sums[k] / acc_counts for k in args.topk}
    return avg_accs, sum(losses) / acc_counts, topk_preds, topk_probs

def Trainer(args, model, loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    acc_sums = {k: 0.0 for k in args.topk}
    acc_counts = 0
    losses = []
    for data in loader:
        x = data.x.to(device) # [batch_size*number_of_nodes, number_of_timesteps] [2048, 48]
        edge_index = data.edge_index.to(device) # [2, batch_size*number_of_edges] [2, 2176]
        y = data.y.to(device) # [batch_size] [64]
        x = x.view(len(y), -1, x.size(-1)) # [batch_size, number_of_nodes, number_of_timesteps] [64, 32, 48]
        out = model(x, edge_index) # [batch_size, number_of_nodes + 1]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc_dict, topk_preds, topk_probs = Utils.accuracy(out, y, topk=args.topk)
        for k in args.topk:
            acc_sums[k] += acc_dict[k]
        
        acc_counts += 1
        losses.append(loss.item())
    
    # Compute average top-k accuracy
    avg_accs = {k: acc_sums[k] / acc_counts for k in args.topk}
    return avg_accs, sum(losses) / acc_counts, topk_preds, topk_probs

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    watergraphdata = WaterGraphDataset(args)
    print(f'idx to node: {watergraphdata.idx_to_node}')
    time.time(100000)
    logging.info(f'Number of nodes: {watergraphdata.number_of_nodes}')
    logging.info(f'Number of edges: {watergraphdata.number_of_edges}')
    torch.manual_seed(args.seed)
    indices = torch.randperm(len(watergraphdata))
    train_indices = indices[:int(0.7 * len(indices))]
    val_indices = indices[int(0.7 * len(indices)) : int(0.8 * len(indices))]
    test_indices = indices[int(0.8 * len(indices)):]
    logging.info(f'Number of training samples: {len(train_indices)}')
    logging.info(f'Number of validation samples: {len(val_indices)}')
    logging.info(f'Number of test samples: {len(test_indices)}')
    train_loader = DataLoader(watergraphdata[train_indices], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(watergraphdata[val_indices], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(watergraphdata[test_indices], batch_size=args.batch_size, shuffle=False)

    model = Model(48, 128, watergraphdata.number_of_nodes+1, 16).to(device)
    history = {'train_loss':[], 'val_loss': [], 'train_top_3': [], 'train_top_5': [], 'val_top_3': [], 'val_top_5': []}
    for epoch in range(args.epoch):
        train_acc_dict, train_loss, _, _ = Trainer(args, model, train_loader, device)
        val_acc_dict, val_loss, _, _ = Tester(args, model, val_loader, device)
        topk_acc_str_train = ' | '.join([f'Top-{k} Acc: {train_acc:6.2f}%' for k, train_acc in train_acc_dict.items()])
        topk_acc_str_val = ' | '.join([f'Top-{k} Acc: {val_acc:6.2f}%' for k, val_acc in val_acc_dict.items()])
        logging.info(f'Epoch: {epoch+1:>3} | Training Loss: {train_loss:.4f} | {topk_acc_str_train} | Val Loss: {val_loss:.4f} | {topk_acc_str_val} ')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_top_3'].append(train_acc_dict[3])
        history['train_top_5'].append(train_acc_dict[5])
        history['val_top_3'].append(val_acc_dict[3])
        history['val_top_5'].append(val_acc_dict[5])

    test_acc_dict, _, topk_preds, topk_probs, y = Tester(args, model, test_loader, device)
    topk_acc_str_test = ' | '.join([f'Top-{k} Acc: {test_acc:6.2f}%' for k, test_acc in test_acc_dict.items()])
    logging.info(f'Test results: {topk_acc_str_test} ')

    if not os.path.exists('./history'):
        os.makedirs('./history')
    with open('./history/GNN.pkl', 'wb') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    args = Configs.parse_args()

    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    Utils.setlogger('./logs/GNN.log')

    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    
    main(args=args)