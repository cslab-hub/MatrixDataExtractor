# backend - Django based Web Application
backend is a web application that provides a basic user interface to store PDF documents on disk and infer document table regions from those PDF documents.
It also allow users to store table data in text files, to transport data from text files to MongoDB and also allow to search data based on Keyword.  

# How to install
The installation process is mentioned on parent README file.

# Pre-requisite

Copy table detection model weight from */tabledetection/diplastmodel/model_final.pth* and paste at */backend/util/data/tabledet/modelweight* folder. The model weight is not provided here. You can use any table detection model weight and relevant config file. You need to rename config file to *uos_dip_config.yaml*. For more 'backend' web application specific setting, check or edit */util/prop/MDE.xml* file.
**NOTE:** Secret Key for Django web application is not provided here, which should be inserted at 'mde.env' file.

# How to run web application
The execution process is mentioned on parent README file.
**Disclaimer:** Other detailed information is mentioned in Di-Plast Wiki page, which is accessable only by project partners.