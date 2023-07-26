FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN pip install jupyter

WORKDIR /workspace

CMD ["bash"]