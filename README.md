# MultiReg
Point Cloud Registration for Multi-objects Matching

## Installation
Use conda to create environment:
```
conda env create -f environment.yml
```
Build dependencies and install the package:
```
python3 setup.py build develop
```

## Demo
### Inference & Visualization
Download pretrained weights
```
wget https://github.com/SheldonFung98/MultiReg/releases/download/V1.0.0/multireg.tar -P weights
```
```
cd demo
python3 inference.py
```
### Model Training
You need to pre-download dataset [ROBI](https://www.trailab.utias.utoronto.ca/robi). The object model in ROBI should be the [reconstructed model](https://drive.google.com/file/d/1mdY9qmlWwYYY4rX7YBWaE1X1tBJDBq7o/view?usp=sharing). 

```text
MultiReg
    |--data
        |--robi
            |--Object_models
                |--Zigzag.stl
            |--Zigzag
                |--Scene_1
            |--Gear
                |--Scene_1
```
Then run the training script
```
cd demo
python3 trainval.py
```