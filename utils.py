import os, json, pickle, glob, re
import numpy as np 
import pandas as pd 
import tarfile
import torch 
from PIL import Image 

from tqdm.auto import tqdm 

from settings import *
import matplotlib.pyplot as plt 

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

def unzip_tarfile(path):
    with tarfile.open(path, 'r') as f:
        f.extractall('dataset')


def save_pkl(df, fname='data/item_info.pkl'):
    if not os.path.exists('data'):
        os.mkdir('data')
        
    with open(fname, 'wb') as f:
        pickle.dump(df, f)
    print(f'Success pickle file, which name is {fname}')
    
    
def load_pkl(fname):
    with open(fname, 'rb') as f:
        files = pickle.load(f)
    return files 


def modify_img_path(path):
    os.chdir(path)
    
    origin_name = glob.glob('*.jpg')
    rename = list(map(lambda x: re.sub('[^a-zA-Z0-9.]', '', x), origin_name)) 
    
    for o_name, r_name in tqdm(zip(origin_name, rename), total=len(rename)):
        os.rename(o_name, r_name)
    os.chdir(BASE_DIR)
    
    print(f'Rename Complete')


def drop_invalid_image(dataframe):
    images, drop_img = [], [] 
    drop_counts = 0.
    target_img = dataframe.photo_id.unique()
    
    for img in tqdm(target_img, total=len(target_img), desc='Loding images'):
        try:
            img = np.array(Image.open(f'{os.path.join(PATH_DICT["image"], img)}'), dtype=np.int8)
            images.append(img)
        except:
            drop_img.append(img)
            drop_counts += 1 
            print(f'The number of drop images is {drop_counts}! Drop image name is {img}')
            
    return drop_img 

def torch2npy(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)
        
    npy = tensor.detach().cpu().numpy()
    return npy

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_imshow(img, reshape=(1, 2, 0), fname='img.pdf', savefig=False):
    if isinstance(img, torch.Tensor):
        img = torch2npy(img)
    img = np.transpose(img, axes=reshape)
    
    plt.figure(figsize = (6, 6))
    plt.imshow(img)
    
    if savefig:
        plt.savefig(f'{fname}.pdf', dpi=100)
    plt.show()

def scaler(x):
    if isinstance(x, torch.Tensor):
        _min = torch.min(x)
        _max = torch.max(x)
        
    elif isinstance(x, np.ndarray):
        _min = np.min(x)
        _max = np.max(x)

    outs = (x - _min) / (_max - _min)
    return outs