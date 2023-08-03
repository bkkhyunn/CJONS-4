import argparse
import torch, time
from tqdm import tqdm 
from torch import nn, optim 
from transformers import BertTokenizer

from utils import * 
from models import MMR, NCF, NCF_LSTM
from data_utils import get_loader 
import wandb 


def evaluate(args, model, valid_loader, criterion):
    valid_loss = 0. 
    with torch.no_grad():
        model.eval()
        for batch, y in valid_loader:
            batch = {k: b.to(args.device) for k, b in batch.items()}
            y = y.to(args.device)
            
            pred_y = model(**batch).squeeze()
            loss = criterion(pred_y, target=y)
        
            valid_loss += loss.item()
        valid_loss /= len(valid_loader)
    return valid_loss 


def train(args, model, train_loader, valid_loader, criterion, optimizer, schedular):
    best_loss = float('inf')
    counts = 0
    for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
        train_loss = 0.
        model.train()
        start = time.time()
        for batch, y in tqdm(train_loader):
            batch = {k: b.to(args.device) for k, b in batch.items()}
            y = y.to(args.device)
            
            pred_y = model(**batch).squeeze()
            loss = criterion(pred_y, target=y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        schedular.step()
        train_loss /= len(train_loader)
        valid_loss = evaluate(args, model, valid_loader, criterion)
        end = time.time()
        elapsed_min, elapsed_sec = elapsed_time(start, end)
        
        if args.wandb:
            wandb.log(
                {'Train Loss': train_loss, 
                 'Valid Loss': valid_loss}
            )
        
        print(f'Train MSE Loss: {train_loss:.4f}\tValid MSE Loss: {valid_loss:.4f}\tElapsed Time: {elapsed_min}m {elapsed_sec:.2f}s')
        
        if best_loss > valid_loss:
            best_loss = valid_loss 
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'{args.model}.pt'))
            counts = 0 
        
        else:
            counts += 1
        
        if counts >= args.patience:
            print('Early Stopping!')
            break 
        
    return train_loss, valid_loss 

MODEL_DICT = {
    'ncf': NCF, 
    'ncf_lstm': NCF_LSTM, 
    'mmr_ad': MMR
}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--num_users', default=11_562, type=int)
    parser.add_argument('--num_items', default=24_033, type=int)
    parser.add_argument('--bidirectional', default=True, action='store_true')
    parser.add_argument('--dr_rate', default=0.2, type=float)
    parser.add_argument('--max_len', default=128, type=int)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--patience', default=3, type=int)
    args = parser.parse_args()
    

    args.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    args.vocab_size = args.tokenizer.vocab_size
     
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    train_df = load_pkl(os.path.join(DATA_DIR, 'train.pkl'))
    valid_df = load_pkl(os.path.join(DATA_DIR, 'valid.pkl'))
    
    print(f'Train dataset: {train_df.shape}\tValid dataset: {valid_df.shape}')
    
    train_loader = get_loader(args, train_df)
    valid_loader = get_loader(args, valid_df)
    
    model = MODEL_DICT[args.model](args).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(args.device)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.001)
    
    if args.wandb:
        wandb.init(
            project='CJons', 
            config={
                'lr': args.lr, 
                'batch_size': args.batch_size, 
                'h_dim': args.hidden_dim, 
                'model': args.model
            }, 
            name=f'{args.model}-lr={args.lr}-h={args.hidden_dim}'
        )
    
    train_loss, valid_loss = train(args, model, train_loader, valid_loader, criterion, optimizer, schedular)