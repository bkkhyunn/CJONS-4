# CJONS-4

#### Description

`data_utils.py`: includes `Dataset`, `DataLoader`.

`models.py`: includes `LSTM`, `NCF`, `ResNet`.

`settings.py`: includes configuration for setting paths.

`utils.py`: includes utilization function.


#### Guide

1. Clone this repository
```
git clone https://github.com/ceo21ckim/CJONS-4.git

cd CJONS-4
```

2. Build Dockerfile
```
docker build --tag [filename]:1.0
```

3. Execute/run docker container
```
docker run -itd --gpus all --name cjons -p 8888:8888 -v C:\[PATH]\:/workspace [filename]:1.0 /bin/bash
```

4. Use jupyter notebook
```
docker exec it [filename] bash

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```
