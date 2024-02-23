# Complete-OCR

Repository contains OCR code for end-to-end outputs from pdfs to hocr outputs of OCR procedure


# Experiments

#### Adobe-API
#### Tabula-Tesseract
#### FRCNN-Tesseract (Current Version)
    Faster rcnn for tables
    Layout parser for images
    Tesseract for text 
    Generates output sets of PDF
    code for it: src/pdf_ocr_frcnn_tesseract_ocr.py
#### Nested-ocr
    DocTR for detection of blocks, lines and words
    Tesseract for multilingual recognition and hocr generation
    Why 'Detection' using DocTR
    Gives nested structure of pages, blocks, lines and words Word level boxes are used to crop page image carry out recognition Nested strucure helps in creating hocrs in concerned format

    Why 'Recognition' using Tesseract
    Availability of multilingual pretrained models Eliminates the computatonal need of NMS or training docTR recog models as of now Open to use other models for recognition stage

    Further utilization
    Results can be exported in the format to be read on Udaan Tool Useful for translation and other downstream applications

    Code for it: src/get_nested_ocr.py


### Installation Steps

```
pip install -r requirements.txt
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
```
