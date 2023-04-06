# Hermes: Client - Server offloading of DNNs


### Prerequisites

Install PyTorch on Jetson using [this](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) tutorial.

### Installation 

```powershell
conda env create --file environment.yml
conda activate hermes
```

If you either added or removed packages, then you can save a checkpoint of the `conda` environment by:

```powershell
conda env export --no-builds > environment.yml
```

### Usage

First, boot up the server:
```powershell
python server.py
```

Then, choose an edge device where PyTorch is installed, and run:
```powershell
python client.py --model resnet18 --image n01667114_mud_turtle.jpg
```

### Dataset

The sample images where downloaded from [this](https://github.com/EliSchwartz/imagenet-sample-images) repo.
