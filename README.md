# IsGCL: Informative Sample-Aware Progressive Graph Contrastive Learning

The code of IsGCL in the unsupervised setting.

## Dependencies

We develop this project with `Python3.7.2, cuda11.0` and  following Python packages:

```
Pytorch                   1.4.0
torch-cluster             1.5.2                    
torch-geometric           1.6.0                    
torch-scatter             2.0.3                    
torch-sparse              0.6.0                    
torch-spline-conv         1.2.0 
```

## Training & Evaluation

### For PisGCL
```
python main.py --DS $DATASET_NAME --aug $AUGMENTATION --beta 0.5 --pacing_type logar --pos_div_threshold 0.01 --neg_div_threshold 0.2 --num_prot 10
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$AUGMENTATION``` could be ```dnodes or pedges```. 

## Acknowledgements

1. The backbone implementation is reference to: https://github.com/ha-lins/PGCL
2. The augmentation implementation is reference to: https://github.com/Shen-Lab/GraphCL.
