from text_attributes import TextAttributes,pdfToimg
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import os

# import pytesseract

import tempfile

pdf_path = "/home/mrportable/Documents/complete-ocr-new/input_books/HO_BR_115_193-1.pdf"

#skip if input is already available as images
image_paths = pdfToimg(pdf_path,"path_to_image_destinations")

img_doc = DocumentFile.from_images(image_paths)
ocr = ocr_predictor(pretrained=True)
out = ocr(img_doc)

result = out.export()
ta = TextAttributes(image_paths, ocr_engine='doctr')
result = ta.generate(result,output_type='json')

print(result)

# def tesseract_multiple_images(images):
#     temp = tempfile.NamedTemporaryFile()
#     with open(temp.name,"w") as f:
#         for image in images:
#             f.write(image+"\n")

#     hocr = pytesseract.image_to_pdf_or_hocr(temp.name, extension='hocr')
#     return hocr

# pdf_path = "./input_books/HO_BR_115_193.pdf"
# image_paths = pdfToimg(pdf_path,"./test/p")
# print(image_paths)
# ta = TextAttributes(image_paths, ocr_engine='tesseract')
# result = tesseract_multiple_images(image_paths)
# result = ta.generate(result,output_type='hocr')
# print(result)

