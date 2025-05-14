import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameters')

    parser.add_argument("--datadir", type=str, default="Dataset", help="data directory")
    parser.add_argument("--network", type=str, default="Hanoi", help="water distribution network")
    parser.add_argument("--sensor_id", type=list, default=['11','13','18','21','24','26','27','3','30','7'], help="selected sensor indices")
    parser.add_argument("--seed", type=int, default=64, help="random seed")
    parser.add_argument("--topk", type=tuple, default=(3,5), help="top k accuracy")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epoch", type=int, default=400, help="the number of training epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
  
    args = parser.parse_args()
    return args