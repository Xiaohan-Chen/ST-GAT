<div align="center">

# Spatio-Temporal Graph Attention Network for Water Distribution Systems

<img height=20 alt="PyG" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"></a>
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<img height=20 alt="PyG" src="https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo_text.svg?sanitize=true"></a>
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

</div>

The rest of the code will be released after the acceptance of the paper.

### Requirements:
- python 3.12.9
- wntr 1.2.0
- torch 2.6.0
- torch-geometric 2.6.1
- numpy 1.26.4
- networkx 3.3

### Dataset:
```bash
tar -xvf Dataset/Hanoi.tar -C Dataset/
```
- 10-sensor position: ['11','13','18','21','24','26','27','3','30','7']
- 8-sensor position: ['13','18','21','24','27','3','30','7']
- 6-sensor position: ['13','21','27','3','30','7']

### Run
```bash
python main.py
```