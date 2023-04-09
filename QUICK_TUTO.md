# Quick tutorial

This tutorial is a quick version of the main tutorial (in `README.md`) except that it is much more condensed and only explains the bare minimum for each command.

![alt text](./docs/logos.png)

Installed versions:

* Python 3.10 and 3.11
* Virtualenv & VirtualenvWrapper
* CUDA 11.8 and CuDNN 8.6 (versions compatible with tensorflow [here](https://www.tensorflow.org/install/source?hl=en#gpu))
* VSCode
* miniconda (optional)

First, uninstall Anaconda if you have it installed. Skip the *CUDA and cuDNN* section if you have not a NVIDIA GPU with compute capability 3.5 or higher (check [this page](https://developer.nvidia.com/cuda-gpus#collapse4) to get compute capability of the device). Basically, a GTX 780 or newer is required.

## Python

Check installed version:

```script
ls /usr/bin/python3*
```

Install Python 3.10:

```script
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10
```

Work also by replacing `python3.10` by `python3.11`.

## Virtualenv & VirtualenvWrapper

Installation:

```script
sudo apt install python3-pip
sudo pip3 install virtualenv virtualenvwrapper
```

Locate the installed files:

```script
which virtualenv
which virtualenvwrapper.sh
```

It should return the same folder called **<virtualenv_dir_path>** in the following.

Create environments and hooks folder:

```script
cd
mkdir venv venv_hooks
```

Update the environment variables (at the end of `~/.bashrc`)

```nano
# virtualenvwrapper
export WORKON_HOME="$HOME/venv"
export VIRTUALENVWRAPPER_PYTHON="/usr/bin/python3"
export VIRTUALENVWRAPPER_HOOK_DIR="$HOME/venv_hooks"
source <virtualenv_dir_path>/virtualenvwrapper.sh
```

Restart the bash.

## CUDA and cuDNN

### Prerequisites

The following lines should return something

```script
# should no return an error:
nvidia-smi
# should no return error (check if nvidia card is present):
lspci | grep -i nvidia
# should return 'x86_64' (architecture)
uname -m
# kernel and gcc versions:
uname -r
gcc --version
```

Kernel and gcc should match the tab: <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements> (same versions or newer)

### CUDA (11.6)

Check current versions:

```script
ls /usr/local/cuda*
```

Find version in the CUDA archives : <https://developer.nvidia.com/cuda-toolkit-archive>. Then click on the button that corresponds to you set-up. And choose **deb (local)** option.

Then follow the instruction to download CUDA (run the command in a folder dedicated to installation to don't loose them).

* Finally, mark the folder as manually installed:

```script
sudo apt-mark manual cuda-\*
```

### cuDNN 8.6

CuDNN page: <https://developer.nvidia.com/cudnn>. Click on "I agree to the terms" and "Archived cuDNN Releases". Look for 8.6 version compatible with CUDA 11.x. Install the Tar file and run:

```script
tar -xvf cudnn-<file_name>.tar.xz
sudo cp cudnn-<file_name>-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-<file_name>-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### Set environment variables

Add the following lines at the end of `~/.bashrc`:

```nano
# cuda path
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

Restart the bash.

### Check installation

```script
mktmpenv -p python3.10
pip install -U pip
pip install tensorflow torch
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

[Tutorial](https://github.com/google/jax#installation) for Jax installation in case of troubles.

In python session:

```python
>>> import tensorflow as tf
>>> import torch
>>> import jax
>>> torch.cuda.is_available()
True
>>> len(tf.config.list_physical_devices('GPU')) > 0
True
>>> len(jax.devices('gpu')) > 0
True
```

## VSCode

Download the .deb file from the website : <https://code.visualstudio.com/download>.

```bash
sudo apt install ./<filename>.deb
code
```

Extensions to install first: *Python*, *Python Indent* and *Pylance*

## Miniconda

### Installation

Go here: <https://docs.conda.io/en/latest/miniconda.html#linux-installers>. Select **Miniconda3 Linux 64-bit**

```script
sh filename.sh
```

### Set-up

Edit `~/.bashrc` by replacing the lines automatically added by Miniconda with:

```nano
# conda
source ~/miniconda3/etc/profile.d/conda.sh
if [[ -z ${CONDA_PREFIX+x} ]]; then
    export PATH="~/conda/bin:$PATH"
fi
```

Deactivate auto-activation of "base" environment:

```script
conda config --set auto_activate_base false
```
