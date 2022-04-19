# Virtualen vs Conda

Conda is a virtual environment manager like Virtualenv and most of online tutorials advice to install it instead of Virtualenv because it is the most simple to use. However Conda have a lot of drawbacks and it is finally less convenient to use by comparison with Virtualenv with its wrapper Virtualenvwrapper even if a quite more complex installation is needed for the later. Fortunately the tutorial provides a complete step-by-step guide to install it properly.

The main drawbacks of Conda and adventages of Virtualenv are:

* Virtualenv only use `pip` for installation where Conda use both `conda` and `pip` that could lead to package conflicts depending on the installation method
* `conda` has difficulties to fix versions conflicts. The pipeline of `pip` is more efficient and it search dirctly versions that match all dependencies
* most developer use only `pip` on their projects so you almost always only use `pip` even in Conda environments (basically when you install packages with: `pip install -r requirements.txt`)
* `conda` has a weird environment architecture. You have a "base" environment automatically activated and all other environment are on a folder inside the folder of "base".
* `conda` has a bad organization of directories and sripts. Each time you create a `conda` environment, a tons of configuration scripts and heavy folders are installed. Only two light folders are installed with Virtualenv: one for the python package and one for binary files (contaning in particular the interpreters)
* Virtualenv has an extension called VirtualenvWrapper that will allows you to have a lot of commands to facilitate the environment management and it will be more convenient that conda even for switching or creating environment (where conda is already easy to use)
* Virtualenv environments are fully compatible with environments created by `venv` command (but more practical to use).
* `conda` is intrusive, it automatically write lines in your `.bashrc` folder to automatically activate the "base" environment when you start a new terminal (and do some other stuff in secret that change your paths). Don't worry, we describe how to avoid problems in the section where we install Miniconda.

The only problem of Virtualenv is that some rare packages are only available with `conda-forge` and so you must work in conda environment if you want them for a project. That's why I'll explain how to install conda in the tutorial. The problems of conda are not sufficient to prevent you from using it quite efficiently sometimes so don't worry about that. **However, when you have the choice always give priority to Virtualenv.**
