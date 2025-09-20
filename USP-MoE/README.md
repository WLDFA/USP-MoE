## Addressing the Challenge of Spatiotemporal Data Scarcity:Cross-City Traffic Flow Prediction Guided by Urban Spatial Patterns


 To address these issues, this paper proposes a cross-city traffic flow prediction method guided by urban spatial patterns, named USP-MoE. The method utilizes a geospatial network to compute the spatial representations of nodes and identifies urban spatial patterns by combining clustering with a traffic flow feedback mechanism. For data-rich source cities, independent expert models are trained for different urban spatial patterns to construct a mixture of experts model aimed at parameter generation. Building on this, the initial parameters of the prediction model for the data-scarce target city are generated based on its specific spatial patterns. These parameters are then fine-tuned through a spatiotemporal graph enhancement strategy to achieve accurate traffic flow prediction and model generalization. 


## Installation
### Environment
- Tested OS: Linux
- Python >= 3.8
- torch == 1.12.0
- torch_geometric == 2.2.0
- Tensorboard

### Dependencies:
1. Install Pytorch with the correct CUDA version.
2. Use the ``pip install -r requirements.txt`` command to install all of the Python modules and packages used in this project.

<!--
## Requirements
- accelerate==0.23.0
- einops==0.7.0
- ema_pytorch==0.2.3
- matplotlib==3.5.3
- numpy==1.23.2
- PyYAML==6.0.1
- PyYAML==6.0.1
- scikit_learn==1.1.2
- scipy==1.9.1
- torch==1.12.0+cu113
- torch_geometric==2.2.0
- torchsummary==1.5.1
- tqdm==4.64.0
- xlrd==2.0.1
- xlwt==1.3.0
-->

## Data
This project uses the original traffic flow data from the LargeST dataset(https://github.com/liuxu77/LargeST.). The dataset contains high-precision traffic flow records from multiple cities, covering both urban roads and county-level areas. It is well-suited for spatiotemporal traffic flow prediction, few-shot learning, and generative prediction model research.

For convenience, the data in this project has been divided by county, and the processed files are stored in the ./Data folder. Each county's data is stored as a separate file with a consistent format, making it easy to load and analyze.



## Model Training

To train node-level models with the traffic dataset, run:

``cd Pretrain``

``CUDA_VISIBLE_DEVICES=0 python pmain.py --taskmode task4 --model v_GWN --test_data metr-la --ifnewname 1 --aftername TrafficData``

After full-trained, run Pretrain\PrepareParams\model2tensor.py to extract parameters from the trained model. And put the params-dataset in ./Data.
 
To train diffusion model and generate the parameters of the target city:

``cd USP-MoE``

``CUDA_VISIBLE_DEVICES=0 python Umain.py --expIndex 140 --targetDataset metr-la --modeldim 512 --diffusionstep 500 --basemodel v_GWN  --denoise Transmoe``


The sample result is in USP-MoE/Output/expXX/.

## Finetune and Evaluate
To finetune the generated parameters of the target city and evaluate, run:

``cd Pretrain``

``CUDA_VISIBLE_DEVICES=0 python pmain.py --taskmode task7 --model v_GWN --test_data metr-la --ifnewname 1 --aftername finetune_7days --epochs 600 --target_days 7``

## Example
If you want to set 'Marin' as target city:

 - In pretrain: You need to merge the data from all counties except Marin to form the source city data, and use the merged source city data as ``test_data``.
 - In Diffusion: set the ``targetDataset`` as 'Marin'.
 - In finetune: set the ``test_dataset`` as 'Marin'.



