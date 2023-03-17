# <div align="center">Structure Consistent Unsupervised Domain Adaptation for Driver Behavior Recognition</div>

## Getting started

* ### Requirements
	<ul>
	<li>pytorch 1.9.1</li>
	<li>torchvision 0.10.1</li>
	<li>wandb 0.12.2</li>
	<li>timm 0.5.5</li>
	<li>prettytable 2.2.0</li>
	<li> scikit-learn </li>
	</ul>
* ### Installation
```
pip install -r requirements.txt
```
We use Weights and Biases ([wandb](https://wandb.ai/site)) to track our experiments and results. To track your experiments with wandb, create a new project with your account. The ```project``` and ```entity``` arguments in ```wandb.init``` must be changed accordingly. To disable wandb tracking, the ```log_results``` flag can be used. 

* ### Datasets
   The datasets used in the repository can be downloaded from the following links:
	   <ul>
	   <li>[VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public) (under classification track)</li>
	   </ul>
	The datasets are automatically downloaded to the ```data/``` folder if it is not available.
## Training
The training scripts can be found under the `examples` subdirectory. 
Sample command to execute the training of the aforementioned methods with a ViT B-16 backbone,  on VisDA dataset can be found below. 
```
python examples/structure_consist_UDA.py /data/VisDA/ -d VisDA2017 -s Synthetic -t Real -a vit_base_patch16_224 --epochs 30 --seed 0 --no-pool --train-resizing cen.crop --log logs/SCUDA_vit/VisDA2017 --log_name VisDA2017_SCUDA_vit --lr 0.002 --log_results -b 64 -j 4
```

## Overview of the arguments
Generally, all scripts in the project take the following flags
- `-a`: Architecture of the backbone. (resnet50|vit_base_patch16_224)
- `-d`: Dataset (VisDA|Driver) 
- `-s`: Source Domain
- `-t`: Target Domain
- `--epochs`: Number of Epochs to be trained for.
- `--no-pool`: Use --no-pool for all experiments with ViT backbone.
- `--log_name`: Name of the run on wandb.
- `--gpu`: GPU id to use.

## Acknowledgement
Our implementation is based on the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library).
