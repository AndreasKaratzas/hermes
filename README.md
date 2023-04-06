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
