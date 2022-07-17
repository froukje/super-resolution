FROM pytorch/torchserve:latest

RUN pip install --no-cache-dir pytorch-lightning
RUN pip install --no-cache-dir h5py
RUN pip install mlflow==1.26.1

COPY ["./model-store", "./model-store"]
COPY ["config.properties", "/home/model-server/config.properies"]

CMD ["torchserve", "--start", "--model-store", "model-store", "--models", "srnet=srnet.mar", "--ts-config", "config.properies"]
