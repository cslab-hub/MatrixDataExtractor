# Di-Plast Matrix Data Extractor
Di-Plast Matrix Data Extractor (MDE) is a web based application, which can be deployed on your computer. It identifies table regions on PDF documents using Computer Vision based Deep Learning, especially Transfer Learning based Object Detection algorithm. Nearly 14 million tons of plastic end up in the ocean every year according to https://www.iucn.org/. A circular economy in plastic industry simultaneously keeps the value of plastics in the economy without leakage into the natural environment. 

Plastic product technical data sheets offer high quality material information commonly in PDF format. Data extraction from such PDF documents is quite complex due to diverse layout and visual appearance of PDF documents. Different plastic product manufacturers follow different types of document templates to provide the relevant information. An information extraction pipeline is essential to integrate such material information into a comprehensive database that can then be leveraged by the stakeholders in the plastic recycling industry. Matrix Data Extractor Tool provides such services leveraging Computer Vision based Object Detection algorithm (Faster R-CNN). Matrix Data Extractor contains two sections- *backend* and *tabledetection*. *tabledetection* is developed on PyTorch 1.8.0 and Detectron2 libraries. Till date Detectron2 library provides official support for Linux OS only, so running *backend* and *tabledetection* without Linux OS is not recommended.

First you need to get Table Detection model weight from */tabledetection/diplastmodel/model_final.pth* and paste at */backend/util/data/tabledet/modelweight* folder. The model weight is not provided here. Also domain specific annotated dataset for table detection is not provided here. If you want, you can use Table Detection benchmark datasets such as PubLayNet, TableBank, ICDAR document table detection datasets for your experiment. The utility functions are provided to convert the dataset from PASCAL-VOC format either to COCO format or to TSV (Tab-Separated Values) format (in .csv files) at */tabledetection/dataprep_util* folder. 

## tabledetection - Deep Learning Model
*tabledetection* is a document table detection framework based on PyTorch and Detectron2 libraries which helps to identify document tables (rectangular boundary box) on PDF documents. It follows Faster R-CNN Object Detection algorithm and uses transfer learning method by incorporating TableBank pre-trained model. The pre-trained model can be downloaded from *Layout-Parser* (https://github.com/Layout-Parser/layout-parser). 

## backend - Web Application
*backend* is a Django based web application which gives a basic user interface to plastic experts to store their PDF files on disk and allows to extract table data by applying Deep Learning and OCR (Optical Character Recognition). It also provide services to store table data in MongoDB and can also provide search functionality of those data using Keywords. Web application specific config file is provided as */backend/util/prop/MDE.xml*.

**NOTE:** Secret Key for Django web application is not provided here, which should be inserted at *mde.env* file. Delete *sample.txt* files in the folders before executing the web application. 

For more details, check README files of *backend* and *tabledetection* sections.

# How to install
Create a conda environment on your Linux OS. Here ENVNAME is considered as **env_mde**. 
```
conda create -n env_mde python=3.8
conda activate env_mde
```

You can follow below link for reference-
- https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf 

To install basic dependencies execute below commands (other versions can work but are not guaranteed to do so)-
```
pip install -r requirements.txt
```
If OpenCV library is not installed properly, try to install by executing below command-
```
pip install opencv-python
```
*tabledetection* model is trained on GPU server and *backend* web application is running on CPU using PyTorch 1.8.0 and relevant Detectron2 library on Linux OS. If you want to train your Deep Learning model, check your CUDA version and install PyTorch 1.8.0 and relevant Detectron2 library. You can get information from below links-
- https://pytorch.org/get-started/previous-versions/
- https://detectron2.readthedocs.io/en/latest/tutorials/install.html

You need to install PyTorch 1.8.0 and Detectron2 CPU version to run *backend* web application on your CPU machine as-
```
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
```
If you are using only *backend* web application, you need to install MongoDB and Elasticsearch on your Linux OS. These are not required for *tabledetection*. You can find below links to install MongoDB and Elasticsearch on your Linux OS-
- https://www.mongodb.com/docs/manual/administration/install-on-linux/
- https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html

You can start and stop MongoDB services on Linux OS by following below commands-
```
sudo systemctl start mongod
sudo systemctl stop mongod
sudo systemctl restart mongod
sudo systemctl status mongod
```
You can start and stop Elasticsearch services on Linux OS by following below commands-
```
sudo service elasticsearch start	
sudo service elasticsearch stop
sudo service elasticsearch restart
sudo service elasticsearch status	
```
You can deactivate the conda environment at the end by executing below command
```
conda deactivate env_mde
```
# Execution

## tabledetection
To train model on Linux OS, execute src/train.py file as-
```
python train.py
```
You can evaluate the model by executing src/test.py file as-
```
python test.py
```
If you are training the model in Slurm mode, you can execute below commnad- 
```
sbatch mde.sh
```
If you are infering your model, you need to create */tabledetection/inferimg/* folder and execute below command-
```
bash -i infer.sh
```
**NOTE:** Make sure all paths to access files and folders are correctly mentioned in corresponding variables in code. You can analyze this code to get an idea.

## backend
Run web application on Linux OS by executing start.sh shell script as-
```
bash -i start.sh 
```
**Disclaimer:** Matrix Data Extractor tool is funded by the Interreg North-West Europe program (Interreg NWE), project Di-Plast - Digital Circular Economy for the Plastics Industry (NWE729, https://www.nweurope.eu/projects/project-search/di-plast-digital-circular-economy-for-the-plastics-industry/). Any support to provide table detection model will not be provided unfortunately after end of project. The accuracy of table detection model depends on various factors such as volume, variety of annotated datasets, hyperparameters of model. You can do your experiment to get better accuracy of your own table detection model. To get table detection model weight on Di-Plast dataset, you can request to Semantic Information Systems Research Group, Osnabrueck University, Osnabrueck, Germany (https://www.informatik.uni-osnabrueck.de/arbeitsgruppen/semantische_informationssysteme.html).
