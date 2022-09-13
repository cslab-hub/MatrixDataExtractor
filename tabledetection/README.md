# tabledetection - Deep Learning model
MDE Table Detection is a document table detection Framework trained on domain specific dataset based on Supervised Learning Object Detection method which uses PyTorch 1.8.0 and Detectron2 library to identify document table (rectangular boundary box) region on plastic product technical datasheets. 

**NOTE:** To increase the accuracy of table detection model, expansion of annotated dataset can be an important factor. You can annotation tool such as LabelImg (https://github.com/tzutalin/labelImg) to annotate document table rectangular boundary box region to create your personal annotated dataset.

# How to install
The installation process is mentioned on parent README file. If you want to export your model (e.g. for C++ developers) using TorchScript. You can install below depencies. Here ENVNAME is considered as **env_mde**.
```
(env_mde)$ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# Table Detection Model Weights and Datasets
The Table Detection model weight and annotated datasets are not provided publicly. You can use document table detection bechmark datasets (e.g. PubLayNet, TableBank, ICDAR table detection datasets) for your research work.

# Pre-trained Model
Pre-trained TableBank model weights and config files are downloaded from *Layout-Parser*. *faster_rcnn_R_101_FPN_3x* is used in Matrix Data Extractor Table Detection. For model weights, go to below link-
- https://github.com/Layout-Parser/layout-parser/blob/main/src/layoutparser/models/detectron2/catalog.py#L36

For config files, go to below link-
- https://github.com/Layout-Parser/layout-parser/blob/main/src/layoutparser/models/detectron2/catalog.py#L62

**NOTE:** The link can be changed if *Layout-Parser* change their code structure.


# Table Detection model - Training and Testing
The training and testing processes are mentioned on parent README file.

## Export Model
Execute below command to export model using TorchScript-
```
$ python src/export_model.py --config-file diplastmodel/faster_rcnn_R_101_FPN_3x_config.yaml \
    --output output --export-method scripting --format torchscript \
    MODEL.WEIGHTS diplastmodel/model_final.pth \
    MODEL.DEVICE cuda
```
## Transfer Learning based Document Table Detection
For more transfer learning based document table detection research work, please check below research paper-

Chowdhury, Arnab Ghosh, Nils Schut, and Martin Atzmüller. "A Hybrid Information Extraction Approach using Transfer Learning on Richly-Structured Documents." LWDA. 2021 (http://ceur-ws.org/Vol-2993/paper-02.pdf).

