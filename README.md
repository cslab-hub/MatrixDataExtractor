# MatrixDataExtractor

MatrixDataExtractor is a web based application that identifies table regions on PDF documents using Computer Vision based Deep Learning algorithms. Nearly 14 million tons of plastic end up in the ocean every year. A circular economy in plastic industry simultaneously keeps the value of plastics in the economy without leakage into the natural environment. 

Plastic product technical data sheets offer high quality material information commonly in PDF format. Data extraction from such PDF documents is quite complex due to diverse layout and visual appearance of PDF documents. Different plastic product manufacturers follow different types of document templates to provide the relevant information. An information extraction pipeline is essential to integrate such material information into a comprehensive database that can then be leveraged by the stakeholders in the plastic recycling industry. MatrixDataExtractor Tool provides such services leveraging Computer Vision based Object Detection algorithms. MatrixDataExtractor contains two sections- backend and tabledetection. 
# Backend
Backend is a Django based web application which gives a basic user interface to plastic experts to store their PDF files on disk and allows to extract table data by applying Deep Learning and OCR (Optical Character Recognition). It also provide services to store table data in MongoDB and can also provide search functionality of those data using Keywords.

# Table Detection
Table Detection is a computer vision based object detection framework based on PyTorch Detectron2 library which helps to identify document table region on PDF documents. 

# How to install



# Tool Structure

