# coac - For specific usecase
A model inference can be performed by executing shell script in a conda environment (*env_mde*) on on Linux CPU machine.

# How to install
The installation process is mentioned on parent README file to create *env_mde* conda environment and install necessary libraries.

# Pre-requisite

- Remove sample.txt files from different directories (e.g. from *coac/data, coac/util/data/inferimgdir, coac/util/data/srcpdf, coac/util/data/tempimgdir, coac/util/data/inferredpdfdir* directories).
- Add annotated dataset in *coac/data* folder (e.g. *coac/data/annotations, coac/data/train, coac/data/val, coac/data/test* directories).
- Add your model weight (e.g. *model_final.pth*) in *coac/diplastmodel* directory.
- Adapt variable values in code according to directory paths.

# Execution
- If you want to create document images from PDF documents, run *pre_process.sh* script. If you want to process single document (not all PDF documents), replace the value of *tdsname* variable before processing.
```
bash -i pre_process.sh
```
- If you want model inference for table detection on extracted document images, run *infer.sh* script. If you want to infer single document (not all PDF documents), replace the value of *tdsname* variable with PDF name before infering. If the folder of *tdsname* value available at *coac/util/data/tempimgdir*, then you will get inferred result at *coac/util/data/inferimgdir* directory.
```
bash -i infer.sh
```
- Bounding box (BBox) information during model inference is stored in *coac/util/bbox* directory in csv files. The data of each csv file is stored into below format
*Filename_PageNo, TABLE_NO, X_MIN, Y_MIN, X_MAX, Y_MAX*, where *PageNo* and *TABLE_NO* is started with index zero, e.g., Filename_0 for first page.

- You can save PDF document with inferred table region at *coac/util/data/inferredpdfdir* before further processing with some tool such as Camelot.

**Disclaimer:** Other detailed information is mentioned in Di-Plast project Wiki page, which is accessable only by project partners.