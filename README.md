# CJONS-4

## Paths

```
├── data
│   ├── data_info.pkl
│   ├── test.pkl
│   ├── train.pkl
│   └── valid.pkl
│
├── dataset
│   ├── photos
│   │   ├── --0h6FMC0V8aMtKQylojEg.jpg
│   │   └── --....jpg
│   │
│   ├── photos.json
│   ├── yelp_academic_dataset_business.json
│   ├── yelp_academic_dataset_review.json
│   ├── yelp_academic_dataset_user.json
│   ├── yelp_dataset.tar
│   └── yelp_photos.tar
│
├── model_parameters
│   ├── ncf.pt
│   ├── ncf_lstm.pt
│   └── mmr_ad.pt
│
├── data_utils.py
├── Dockerfile
├── models.py
├── settings.py
├── utils.py
└── README.md
```


## Description

We're providing guidelines for Multi-Modal Recommender Systems with Anomaly Detection (For short MMR-AD) that is proposed by `CJons-4 Team` based on the datasets available at [Yelp.com](https://www.yelp.com/dataset). We implemented MMR-AD by using `PyTorch`, `Scikit-learn`, `Pandas`, `etc`.

`data_utils.py`: includes `Dataset`, `DataLoader`.

`models.py`: includes `LSTM`, `NCF`, `ResNet`.

`settings.py`: includes configuration for setting paths.

`utils.py`: includes utilization function.

## Unzip tarfile
```
from settings import * 
import tarfile, glob 

def unzip_tarfile(path):
    with tarfile.open(path, 'r') as f:
        f.extractall('dataset')
        
paths = glob.glob(DATA_DIR + '/*.tar')

for p in paths:
    unzip_tarfile(p)

```

## Guide

**1. Clone this repository**
```
git clone https://github.com/ceo21ckim/CJONS-4.git

cd CJONS-4
```

**2. Build Dockerfile**
```
docker build --tag [filename]:1.0
```

**3. Execute/run docker container**
```
docker run -itd --gpus all --name cjons -p 8888:8888 -v C:\[PATH]\:/workspace [filename]:1.0 /bin/bash
```

**4. Use jupyter notebook**
```
docker exec -it [filename] bash

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```


## Training

![image](https://github.com/ceo21ckim/CJONS-4/blob/main/asset/image1.PNG)
