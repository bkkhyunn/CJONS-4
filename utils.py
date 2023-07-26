import os, json
import numpy as np 
import pandas as pd 
import torch 
from tqdm.auto import tqdm 

def txt2str(path):
    with open(os.path.join(path)) as f:
        outs = f.readline()
    return outs 

def accuracy(pred_y, target):
    pred_y = pred_y.detach()
    return ((pred_y > 0.5) == target).sum() / len(target)

def elapsed_time(start, end):
    elapsed = end - start 
    elapsed_min = elapsed // 60 
    elapsed_sec = elapsed - elapsed_min * 60 
    return elapsed_min, np.round(elapsed_sec, 2)

def json2df(path):
    json_data = []
    with open(path, 'r') as f:
        for line in tqdm(f, desc='transformating json to dataframe...'):
            json_data.append(json.loads(line))
    return pd.json_normalize(json_data)



# def evaluate(args, model, valid_loader, criterion):
#     valid_loss, valid_acc = 0., 0.
#     with torch.no_grad():
#         model.eval()
#         for batch in valid_loader:
#             X, y = batch['text'].to(args.device), batch['label'].to(args.device)

#             logits, pred_y = model(X.squeeze())
#             loss = criterion(logits, target=y)
#             acc = accuracy(pred_y, target=y)
#             valid_loss += loss.item()
#             valid_acc += acc.item()
#         valid_loss /= len(valid_loader)
#         valid_acc /= len(valid_loader)
    
#     return valid_loss, valid_acc


# def train(args, model, train_loader, valid_loader, criterion, optimizer):
#     for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
#         best_loss, best_acc = float('inf'), float('-inf')

#         train_loss, train_acc = 0., 0.
#         model.train()
#         for batch in train_loader:
#             X, y = batch['text'].to(args.device), batch['label'].to(args.device)
#             logits, pred_y = model(X.squeeze())
#             loss = criterion(logits, target=y)
#             acc = accuracy(pred_y, target=y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             train_acc += acc.item()
#         train_loss /= len(train_loader)
#         train_acc /= len(train_loader)
#         valid_loss, valid_acc = evaluate(args, model, valid_loader, criterion)

#         print(f'Epoch: [{epoch+1}/{args.num_epochs}]')
#         print(f'Train Loss: {train_loss:.4f}\tValid Loss: {valid_loss:.4f}')
#         print(f'Train Accuracy: {train_acc*100:.2f}\tValid Accuracy: {valid_acc*100:.2f}')

#     if best_loss > valid_loss:
#         best_loss = valid_loss 
#         torch.save(model.state_dict(), 'model_parameters.pt')

#     return {
#         'train_loss': train_loss, 
#         'valid_loss': valid_loss
#     }