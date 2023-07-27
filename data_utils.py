import os, torch 

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms 
from collections import defaultdict
from tqdm.auto import tqdm 
from PIL import Image 

from settings import * 
from utils import scaler


class Dset(Dataset):
    def __init__(self, args, dataframe):
        self.df = dataframe
        self.size = args.size
        self.max_len = args.max_len
        self.user_ids = self.df.user_id.values
        self.item_ids = self.df.business_id.values
        self.reviews = self.df.text.values
        self.ratings = scaler(self.df.stars.values)
        self.tokenizer = args.tokenizer
        self.images = self.df.loc[:, ['business_id', 'photo_id']].drop_duplicates().values
        
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)), 
            transforms.ToTensor(), 
            transforms.Normalize(0.5, 0.5), 
        ])
        
        image = defaultdict(list)
        for (item_id, image_id) in tqdm(self.images, total=len(self.images), desc='Loading images...'):
            img = Image.open(f'{os.path.join(PATH_DICT["image"], image_id)}')
            image[item_id] = img 
        
        self.images = image

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user = self.user_ids[idx]
        item = self.item_ids[idx]
        rating = self.ratings[idx]
        
        review = self.reviews[idx]
        review = self.tokenizer.encode_plus(
            text=review, 
            add_special_tokens=False, 
            max_length=self.max_len,
            padding='max_length', 
            truncation=True 
        )['input_ids']
        
        img = self.images[item]
        img = self.transform(img)

        return {
            'user': torch.tensor(user, dtype=torch.long), 
            'item': torch.tensor(item, dtype=torch.long), 
            'review': torch.tensor(review, dtype=torch.long),
            'image': img, 
        }, torch.tensor(rating, dtype=torch.float32)
    
def get_loader(args, dataframe):
    d_set = Dset(args, dataframe)
    return DataLoader(d_set, batch_size=args.batch_size)