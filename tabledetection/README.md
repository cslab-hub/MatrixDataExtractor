# tabledetection - Deep Learning model
MDE Table Detection is a document table detection Framework trained on domain spefici dataset which uses PyTorch 1.8.0 and Detectron2 library to identify document table (rectangular boundary box) region on plastic product technical datasheets. 

# How to install
The installation process is mentioned on parent README file. If you want to export your model (e.g. for C++ developers) using TorchScript. You can install below depencies-
```
(env_mde)$ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

# Table Detection Model Weights and Datasets
The Table Detection model weight and annotated datasets are not provided publicly. You can use document table detection bechmark datasets (e.g. PubLayNet, TableBank, ICDAR table detection datasets) for your research work.

# Pre-trained Model
Pre-trained TableBank model weights and config files are downloaded from Layout-Parser. 'faster_rcnn_R_101_FPN_3x' is used in Matrix Data Extractor Table Detection. For model weights, go to below link-
- https://github.com/Layout-Parser/layout-parser/blob/main/src/layoutparser/models/detectron2/catalog.py#L36

For config files, go to below link-
- https://github.com/Layout-Parser/layout-parser/blob/main/src/layoutparser/models/detectron2/catalog.py#L62

**NOTE:** The link can be changed if Layout-Parser change their code structure.


# Table Detection model - Training and Testing
The trainign and testing processes are mentioned on parent README file.

## Export Model
Execute below command to export model using TorchScript-
```
$ python src/export_model.py --config-file diplastmodel/uos_dip_config.yaml \
    --output output --export-method scripting --format torchscript \
    MODEL.WEIGHTS diplastmodel/model_final.pth \
    MODEL.DEVICE cuda
```
