# Assignment 0 - Python/Jupyter/Gradescope

This assignment is designed to help you prepare your local python environment, introduce you to jupyter notebooks and provide a refresher on python language. After following this README you will have a python environment ready and will be able to proceed with learning about jupyter notebooks in the `notebook.ipynb` (where you will make your first graded submission!). Let's get started!

### Table of Contents
- [Get repository](#repo)
- [Conda](#conda)
- [Environment](#env)
- [Packages](#pkg)
- [Jupyter](#jupyter)
- [Summary](#summary)

<a name="repo"/></a>
## Get repository

First things first, let's pull this repository to your local machine:

```
git clone https://github.gatech.edu/omscs6601/assignment_0.git
```

Then come back to this README to continue with further setup.

<a name="conda"/></a>
## Conda

![Conda Logo](https://conda.io/en/latest/_images/conda_logo.svg)

Conda is an open source package and environment management system. Conda quickly installs, runs and updates packages/libraries and easily creates, saves, loads, and switches between environments on your local computer.

Please download [Miniconda (Python3.7)](https://docs.conda.io/en/latest/miniconda.html) and install it on your local machine. You can access conda via the console, to make sure it's properly installed please run `conda -V` to display the version.

On Windows, to access `conda` via the console please use "Anaconda Prompt" or "Anaconda Powershell Prompt" instead of "Command Prompt".

<a name="env"/></a>
## Environment

Environments are used to keep different python versions and packages isolated from each other, generally each project/application will have an independent python environment. For example, we will be using Python 3.7 and packages like numpy, networkx etc, and we want them to be isolated from any other python projects you might have. 

To create a new environment simply run:

```
conda create --name ai_env python=3.7 -y
```

Once it's created you can activate it by running:

```
conda activate ai_env
```

The environment is not attached to any specific folder, you can freely navigate to different directories while it's activated. If you want to change the environment you can deactivate it using `conda deactivate` and then activate another env. To see the list of all environments you have on your machine you can run `conda env list`.

<a name="pkg"/></a>
## Packages

![Python Logo](https://www.python.org/static/community_logos/python-logo-master-v3-TM.png)

We will be using multiple python packages throughtout this class, here are some of them:

* **jupyter** - interactive notebook (you will learn more about them soon)
* **numpy** - a package for scientific computing (multi-dimensional array manipulation)
* **matplotlib** - a plotting library
* **networkx** - a package for manipulating networks/graphs
* **pandas** - a package for data analysis
* **pgmpy** - library for probabilistic graphical models 

You can see the complete list of packages and required versions in [./requirements.txt](./requirements.txt).

We can install all these packages using command ``pip install -r requirements.txt``. Please navigate to `assignment_0/` directory, activate your environment (`conda activate ai_env`), then run:

```
pip install -r requirements.txt
```

Once installed, you can run `pip freeze` to see the list of all of the packages installed in your `ai_env` environment.

<a name="jupyter"/></a>
## Jupyter

![Jupyter Logo](https://jupyter.org/assets/nav_logo.svg)

Now that you have setup the environment it's time to learn more about the jupyter notebooks. 

We have already installed jupyter, to open it up you can run:

```
jupyter notebook
```

It will start a python kernel which you can assess via [https://localhost:8888](https://localhost:8888/) in your browser, for the rest of the assignment proceed to `notebook.ipynb`.

<a name="summary"/></a>
## Summary

You have installed conda package and environment manager, created a python environment and installed all the necessary packages.

Please always remember to run:
```
conda activate ai_env
```
to activate your environment before you start working on the assignment.