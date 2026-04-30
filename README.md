## Towards Discriminative Multimodal Entity Linking for Ambiguous Entities via Hard Negative Clustering

## Dependencies
* conda create -n dismel python=3.7 -y
* torch==1.11.0+cu113
* transformers==4.27.1
* torchmetrics==0.11.0
* tokenizers==0.12.1
* pytorch-lightning==1.7.7
* omegaconf==2.2.3
* pillow==9.3.0


## Running the code
### Dataset
1. Download the datasets from [MIMIC](https://github.com/pengfei-luo/MIMIC).
2. Download the datasets processed with hard-negative clustering, which is provided in this project at: `./datasExample/`.
 Alternatively, you can download the datasets with WikiData description information from the [MMoE](https://drive.google.com/drive/folders/196zSJCy5XOmRZ995Y1SUZkGbMN922nPY?usp=sharing) and generate the processed datasets by running `preprocessing/xxx/pre4obj.py`, `preprocessing/xxx/rank_nn.py` and `preprocessing/xxx/cluster.py`.
 Then move it to the corresponding MIMIC datasets folder.
3. Create the data root directory, move the datasets into it, and update `data.root` in `./config/xxx.yaml` accordingly.


### Training model
```python
python main.py --config config/xxx.yaml --m record_info
```

### Training logs
**Note:** We provide logs of our training in the logs directory.

