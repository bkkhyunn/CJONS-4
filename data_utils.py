from torch.utils.data import DataLoader, Dataset

class Dset(Dataset):
    def __init__(self, args, dataframe):
        self.df = dataframe 
        self.user_ids = self.df.user_id
        self.item_ids = self.df.item_id 
        self.photos = self.df.photo
        self.reviews = self.df.review
        self.ratings = self.df.rating
        self.tokenizer = args.tokenizer
        raise NotImplementedError

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        raise NotImplementedError
    

def get_loader(args, dataframe):
    d_set = Dset(args, dataframe)
    return DataLoader(d_set, batch_size=args.batch_size)



# class IMDB(Dataset):
#     def __init__(self, args, paths):
#         self.paths = paths 
#         self.tokenizer = args.tokenizer
#         self.txts =[]

#         self.labels = [1. if 'pos' in path else 0. for path in self.paths]

#         for path in tqdm(self.paths):
#             sentence = txt2str(path)
#             sentence = self.tokenizer.encode_plus(
#                 text=sentence, 
#                 add_special_tokens=True,
#                 max_length=args.max_len,
#                 padding='max_length', 
#                 truncation=True
#             )

#             self.txts.append([sentence['input_ids']])
#         print(f'POS Label: {sum(self.labels):,}\tNEG Label: {len(self.labels) - sum(self.labels):,}')
        
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):

#         return {
#             'text': torch.tensor(self.txts[idx], dtype=torch.long), 
#             'label': torch.tensor(self.labels[idx], dtype=torch.float32)
#         }