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


def conv3x3(in_dim, out_dim, stride = 1, padding=1):
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_dim, out_dim, stride = 1):
    return nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, in_dim, out_dim, stride = 1, padding=1):
        super(Bottleneck, self).__init__()
        norm_layer = nn.BatchNorm2d
        
        self.conv1 = conv1x1(in_dim, out_dim)
        self.bn1 = norm_layer(out_dim)
        
        self.conv2 = conv3x3(out_dim, out_dim, stride, padding)
        self.bn2 = norm_layer(out_dim)
        
        self.conv3 = conv1x1(out_dim, out_dim * 2)
        self.bn3 = norm_layer(out_dim * 2)
        
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, args, in_dim, out_dim, num_layers):
       self.in_dim = in_dim
       self.out_dim = out_dim 
       self.num_layers = num_layers 
       self.args = args 
       
       self.block = nn.Sequential(
           Bottleneck(in_dim, out_dim), 
           Bottleneck(out_dim, out_dim * 2)
       )
       
       self.fc_layer = nn.Sequential(
           
       )


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