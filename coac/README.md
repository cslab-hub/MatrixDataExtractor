# coac - For specific usecase
A model inference can be performed by executing shell script in a conda environment (*env_mde*) on on Linux CPU machine.

# How to install
The installation process is mentioned on parent README file to create *env_mde* conda environment and install necessary libraries.

# Pre-requisite

- Remove sample.txt files from different directories (e.g. from *data, /util/data/inferimgdir, /util/data/srcpdf, /util/data/tempimgdir* directories).
- Add annotated dataset in *data* folder (e.g. */data/annotations, /data/train, /data/val, /data/test* directories).
- Add your model weight (e.g. *model_final.pth*) in *diplastmodel* directory.
- Adapt variable values in code according to directory paths.

# Execution
- If you want to create document images from PDF documents, run *pre_process.sh* script.
```
bash -i pre_process.sh
```
- If you want model inference for table detection on extracted document images, run *infer.sh* script.
```
bash -i infer.sh
```
- Bounding box (BBox) information during model inference is stored in */util/prop/bbox_info.csv* file. The data in this csv file is stored into below format
*Filename_PageNo, TABLE_NO, X_MIN, Y_MIN, X_MAX, Y_MAX*, where *PageNo* and *TABLE_NO* is started with index zero, not one.

# Execute Streamlit Application
- If you want to execute streamlit application to extract table structure in an interactive mode, run below command
```
streamlit run src/app.py
```
**Disclaimer:** Other detailed information is mentioned in Di-Plast Wiki page, which is accessable only by project partners.