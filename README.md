# FreezeNet: A lightweight deep learning model for enhancing freeze tolerance assessment and genetic analysis in wheat
## Description
### Model
FreezeNet, a lightweight neural network, is suitable for image segmentation of wheat seedlings in field environments.


## Requirements
Install dependencies using pip
```
pip install -r requirements.txt
```

## Train:
`train.py` is used to train segmentation models
```bash
python train.py
```
## Predict
`predict.py` is used to segment wheat plants from images
```bash
python predict.py
```
## Phenotyping:
The phenotyping method is in the `phenotypying.ipynb` 

Green leaves and yellow leaves segmentation
![LeavesSeg](./assets/seg.gif)





