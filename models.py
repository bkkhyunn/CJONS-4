import torch
from torch import nn



class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.bi = args.bidirectional
        self.dropout = nn.Dropout(args.dr_rate)
        self.txt_emb = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.lstm = nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim//4, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        embs = self.txt_emb(inputs)
        o, (h, c) = self.lstm(embs)
        h = self.dropout(h)
        
        if self.bi:
            h = torch.cat([h[-1], h[-2]], dim=-1)
        
        else:
            h = h[-1]

        return h
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


class UserItem(nn.Module):
    def __init__(self, args):
        super(UserItem, self).__init__()

        self.num_users = args.num_users 
        self.num_items = args.num_items 
        self.hidden_dim = args.hidden_dim
        
        self.user_emb = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_emb = nn.Embedding(self.num_items, self.hidden_dim)

    def forward(self, user_id, item_id):
        user_emb = self.user_emb(user_id)
        item_emb = self.item_emb(item_id)

        outs = torch.concat([user_emb, item_emb], dim=-1)

        return outs 



class CNNBlock(nn.Module):
    def __init__(self, args):
       super(CNNBlock, self).__init__()
       self.size = args.size
       self.args = args
       self.hidden_dim = args.hidden_dim
       self.pooling = 2
       self.num_layers = [3, 8, 16, 32, 64, 128]
       self.layers = nn.Sequential()
       
       for idx, (in_dim, out_dim) in enumerate(zip(self.num_layers[:-1], self.num_layers[1:])):
           self.layers.add_module(f'cnn_block-{idx+1}', self.block3x3(in_dim=in_dim, out_dim=out_dim, activation='relu', pooling=self.pooling))

       self.fc_layer = nn.Sequential(
           nn.Linear( args.size * args.size * self.num_layers[-1] // (self.pooling ** (len(self.num_layers)-1))**2, self.hidden_dim)
       )
       
    def block3x3(self, in_dim=3, out_dim=8, activation='relu', pooling=False):
        layers = nn.Sequential(
            self.conv3x3(in_dim, out_dim), 
            nn.BatchNorm2d(out_dim)
        )
        if activation == 'relu':
            layers.add_module('relu', nn.ReLU())
        
        elif activation == 'softmax':
            layers.add_module('softmax', nn.Softmax())
        
        if pooling:
            layers.add_module('pooling', nn.MaxPool2d(pooling))
        
        return layers 
    
    def forward(self, img):
        outs = self.layers(img)
        outs = self.fc_layer(outs.flatten(1))
        return outs
    
    def conv3x3(self, in_dim, out_dim, stride = 1, padding=1):
        return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding, bias=False)

    def conv1x1(self, in_dim, out_dim, stride = 1):
        return nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)
    

# Proposed Model: user_id + item_id + reviews + images
class MMR_AD(nn.Module):
    def __init__(self, args):
        super(MMR_AD, self).__init__()
        self.ncf = UserItem(args)
        self.lstm = LSTM(args)
        self.resnet = CNNBlock(args)

        self.hidden_dim = args.hidden_dim

        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * (2 + 1) + self.hidden_dim // 2, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item, review, image):

        ui_emb = self.ncf(user, item) # (b, hidden_dim * 2) 128
        review_emb = self.lstm(review) # (b, hidden_dim // 2) 32
        img_emb = self.resnet(image) # (b, hidden_dim)

        outs = torch.concat([ui_emb, review_emb], dim=-1)
        outs = torch.concat([outs, img_emb], dim=-1)

        outs = self.fc_layer(outs)
        outs = self.sigmoid(outs)
        return outs 


# Baseline Model: user_id + item_id + reviews
class NCF_LSTM(nn.Module):
    def __init__(self, args):
        super(NCF_LSTM, self).__init__()
        self.user_emb = nn.Embedding(args.num_users, args.hidden_dim)
        self.item_emb = nn.Embedding(args.num_items, args.hidden_dim)
        self.lstm = LSTM(args)
        
        self.hidden_dim = args.hidden_dim
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.hidden_dim // 2, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user, item, review):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        
        ui_emb = torch.concat([user_emb, item_emb], dim=-1)
        review_emb = self.lstm(review) # (b, hidden_dim // 2) 32

        outs = torch.concat([ui_emb, review_emb], dim=-1)

        outs = self.fc_layer(outs)
        outs = self.sigmoid(outs)
        return outs 
    
    
# Baseline Model: user_id + item_id 
class NCF(nn.Module):
    def __init__(self, args):
        super(NCF, self).__init__()

        self.num_users = args.num_users 
        self.num_items = args.num_items 
        self.hidden_dim = args.hidden_dim
        
        self.user_emb = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_emb = nn.Embedding(self.num_items, self.hidden_dim)
        
        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim//2, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)

        outs = torch.concat([user_emb, item_emb], dim=-1)
        outs = self.fc_layer(outs)
        outs = self.sigmoid(outs)
    
        return outs