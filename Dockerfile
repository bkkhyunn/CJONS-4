FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN pip install transformer==4.24.0 && tqdm && pandas && scikit-learn \
    && opencv-python && matplotlib && fasttext && mxnet && gluonlp && boto3 && botocore

RUN pip install jupyter

WORKDIR /workspace

CMD ["bash"]