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
    def __init__(self, args):
        super(NCF, self).__init__()

        self.num_users = args.num_users 
        self.num_items = args.num_items 
        self.hidden_dim = args.hidden_dim
        
        self.user_emb = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_emb = nn.Embedding(self.num_items, self.hidden_dim)

        self.layers = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim//2), 
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
        
        self.conv1 = conv3x3(in_dim, out_dim)
        self.bn1 = norm_layer(out_dim)        
        
        self.conv2 = conv3x3(out_dim, out_dim)
        self.bn2 = norm_layer(out_dim)
        
        self.conv3 = conv3x3(out_dim, out_dim * 2)
        self.bn3 = norm_layer(out_dim * 2)
        
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, args):
       super(ResNet, self).__init__()
       self.in_dim = 3
       self.out_dim = args.hidden_dim // 2
       self.args = args 
       
       self.block = nn.Sequential(
           Bottleneck(self.in_dim, self.out_dim), 
           Bottleneck(self.out_dim * 2, self.out_dim * 2)
       )
       
       self.fc_layer = nn.Sequential(
           nn.Linear(256 * 256 * self.out_dim * 4, 64)
       )
       
    def forward(self, img):
        outs = self.block(img)
        outs = self.fc_layer(outs.flatten(1))
        return outs
    
    
class MMR_AD(nn.Module):
    def __init__(self, args):
        super(MMR_AD, self).__init__()
        self.ncf = NCF(args)
        self.lstm = LSTM(args)
        self.resnet = ResNet(args)

        self.hidden_dim = args.hidden_dim

        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim, self.hidden_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.hidden_dim // 2, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item, review, image):

        ui_emb = self.ncf(user, item) # (h, hidden_dim // 4) 16
        review_emb = self.lstm(review) # (b, hidden_dim // 4) 16
        img_emb = self.resnet(image) # (b, hidden_dim)

        outs = torch.concat([ui_emb, review_emb], dim=-1)
        outs = torch.concat([outs, img_emb], dim=-1)

        outs = self.fc_layer(outs)
        outs = self.sigmoid(outs)
        return outs 