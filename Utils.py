import logging
import torch.nn.functional as F
import torch
import time

def setlogger(path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(filename=path)

    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    prob = F.softmax(output, dim=1) # convert logits to probabilities

    # prob = output
    topk_prob, topk_idx = prob.topk(maxk, dim=1, largest=True, sorted=True) # [batch_size, maxk]

    pred = topk_idx.t() # [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # [maxk, batch_size]
    
    accs = {}
    topk_preds = {}
    topk_probs = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1,).float().sum(0, keepdim=True)
        accs[k] = (correct_k * (100.0 / batch_size)).cpu().numpy().item()
        topk_preds[k] = topk_idx[:, :k]
        topk_probs[k] = topk_prob[:, :k]
    return accs, topk_preds, topk_probs