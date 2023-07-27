import argparse
import torch, time
from tqdm import tqdm 
from torch import nn, optim 
from transformers import BertTokenizer

from utils import * 
from models import MMR_AD
from data_utils import get_loader 

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=16, type=16)
parser.add_argument('--hidden_dim', default=64, type=int)
parser.add_argument('--num_users', default=56_432, type=int)
parser.add_argument('--num_items', default=25_328, type=int)
parser.add_argument('--bidirectional', default=True, action='store_true')
parser.add_argument('--dr_rate', default=0.2, type=float)
parser.add_argument('--max_len', default=128, type=int)
parser.add_argument('--size', default=128, type=int)

args = parser.parse_args()


def evaluate(args, model, valid_loader, criterion):
    valid_loss = 0. 
    with torch.no_grad():
        model.eval()
        for batch, y in valid_loader:
            batch = {k: b.to(args.device) for k, b in batch.items()}
            y = y.to(args.device)
            
            pred_y = model(**batch)
            loss = criterion(pred_y, target=y)
        
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)
    return valid_loss 


def train(args, model, train_loader, valid_loader, criterion, optimizer, schedular):
    best_loss = float('inf')
    for epoch in tqdm(args.num_epochs, total=args.num_epochs):
        train_loss = 0.
        model.train()
        start = time.time()
        for batch, y in train_loader:
            batch = {k: b.to(args.device) for k, b in batch.items()}
            y = y.to(args.device)
            
            pred_y = model(**batch)
            loss = criterion(pred_y, target=y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.zero_grad()
        schedular.step()
        train_loss /= len(train_loader)
        valid_loss = evaluate(model, valid_loader, criterion)
        end = time.time()
        elapsed_min, elapsed_sec = elapsed_time(start, end)
        
        print(f'Train MSE Loss: {train_loss:.4f}\tValid MSE Loss: {valid_loss:.4f}\tElapsed Time: {elapsed_min}m {elapsed_sec:.2f}s')
        
        if best_loss > valid_loss:
            best_loss = valid_loss 
            torch.save(model.state_dict(), SAVE_DIR)
        
    return train_loss, valid_loss 


if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
     
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    train_df = load_pkl(os.path.join(DATA_DIR, 'train.pkl'))
    valid_df = load_pkl(os.path.join(DATA_DIR, 'valid.pkl'))
    
    train_loader = get_loader(args, train_df)
    valid_loader = get_loader(args, valid_df)
    
    model = MMR_AD(args).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(args.device)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.001)