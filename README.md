# Hermes: Client - Server offloading of DNNs


### Prerequisites

Install PyTorch on OrangePi 5 using [this](https://pytorch.org/tutorials/intermediate/realtime_rpi.html) tutorial.

### Installation 

On the server, run:

```powershell
conda env create --file environment.yml
conda activate hermes
```

If you either added or removed packages, then you can save a checkpoint of the `conda` environment by:

```powershell
conda env export --no-builds > environment.yml
```

On the client, on a Python 3.10.6 environment run:
```powershell
pip install torch torchvision torchaudio
```

### Usage

* Demo

First, boot up the server:
```powershell
python server.py
```

Then, choose an edge device where PyTorch is installed, and run:
```powershell
python client.py --model resnet18 --image n01667114_mud_turtle.jpg
```

* Real example:

You may need to allow TCP traffic on a port:
```powershell
sudo ufw allow 8080
```

Then, start the server:
```powershell
python server.py --ip 0.0.0.0 --port 8080
```

Finally, you can perform tests through the client:
```powershell
python client.py --model resnet18 --image n01667114_mud_turtle.jpg --ip 131.230.193.241 --port 8080
```

### Dataset

The sample images where downloaded from [this](https://github.com/EliSchwartz/imagenet-sample-images) repo.
