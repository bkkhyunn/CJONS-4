import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.bi = args.bidirectional
        self.dropout = nn.Dropout(self.dr_rate)
        self.txt_emb = nn.Embedding(args.vocab_size, args.hidden_dim)
        self.lstm = nn.LSTM(input_size=args.hidden_dim, hidden_size=args.hidden_dim//4, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        embs = self.txt_emb(inputs)
        o, (h, c) = self.lstm(embs)
        
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


class NCF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(NCF, self).__init__()

        self.num_users = num_users 
        self.num_items = num_items 
        self.hidden_dim = args.hidden_dim
        
        self.user_emb = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_emb = nn.Embedding(self.num_items, self.hidden_dim)

        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim//2, self.hidden_dim//4)
        )


    def forward(self, user_id, item_id):
        user_emb = self.user_emb(user_id)
        item_emb = self.item_emb(item_id)

        inputs = torch.concat([user_emb, item_emb], dim=-1)
        outs = self.layers(inputs)

        return outs 
    
# class MMR_AD(nn.Module):
#     def __init__(self, args):
#         self.ncf = NCF(args)
#         self.lstm = LSTM(args)
#         self.resnet = ResNet(args)

#         self.hidden_dim = args.hidden_dim

#         self.fc_layer = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.hidden_dim//2), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim//2, self.hidden_dim//4), 
#             nn.ReLU(), 
#             nn.Linear(self.hidden_dim//4, 1)
#         )

#         self.sigmoid = nn.Sigmoid()


#     def forward(self, user_id, item_id, review, image):

#         ui_emb = self.ncf(user_id, item_id)
#         review_emb = self.lstm(review)
#         img_emb = self.resnet(image)

#         outs = torch.concat([ui_emb, review_emb], dim=-1)
#         outs = torch.concat([outs, img_emb], dim=-1)

#         outs = self.fc_layer(outs)
#         outs = self.sigmoid(outs)
#         return outs 