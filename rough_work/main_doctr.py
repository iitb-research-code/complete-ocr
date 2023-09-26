# Import Libraries
import os
import time

import shutil
import argparse

import torch
import pytesseract

from pdf2image import convert_from_path
from PIL import Image

from bs4 import BeautifulSoup
import cv2

import json

# use torch environment (lpmodel is using tesonflow so disable)
os.environ["USE_TORCH"] = "1"

import warnings
warnings.filterwarnings("ignore")

# Importing DocTR libraries
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from doctr.io import DocumentFile
from doctr.models import crnn_vgg16_bn, db_resnet50
from doctr.models.predictor import OCRPredictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor

from text_attributes import TextAttributes

from utility.config import *
from figureDetection import *




### For parsing boolean from string
def parse_boolean(b):
    return b == True


### For simple filename generation
def simple_counter_generator(prefix="", suffix=""):
    i=0
    while True:
        i+=1
        yield 'p'


### Initializing models for handwritten OCR
def initialize_handwritten_models(language_model):
    if(MODEL[language_model] =='default'):
        predictor = ocr_predictor(pretrained=True)
    else:
        det_model = db_resnet50(pretrained=True)
        det_predictor = DetectionPredictor(PreProcessor((1024, 1024), batch_size=1, mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)), det_model)

        #Recognition model
        reco_model = crnn_vgg16_bn(pretrained=False, vocab=VOCABS[MODEL[language_model]])
        reco_param = torch.load(REC_MODEL_PATH + MODEL[language_model] +'.pt', map_location="cpu")
        reco_model.load_state_dict(reco_param)
        reco_predictor = RecognitionPredictor(PreProcessor((32, 128), preserve_aspect_ratio=True, batch_size=1, mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301)), reco_model)

        predictor = OCRPredictor(det_predictor, reco_predictor)

    return predictor


def get_hocr(json_data):

    hocr_content = []
    for page in json_data['pages']:
        hocr_content.append('<div class="ocr_page">')
        for block in page['blocks']:
            block_geometry = block['geometry']
            hocr_content.append(f'<p class="ocr_carea" title="bbox {block_geometry[0][0]} {block_geometry[0][1]} {block_geometry[1][0]} {block_geometry[1][1]}">')
            for line in block['lines']:
                line_geometry = line['geometry']
                hocr_content.append(f'<span class="ocr_line" title="bbox {line_geometry[0][0]} {line_geometry[0][1]} {line_geometry[1][0]} {line_geometry[1][1]}">')
                for word in line['words']:
                    word_geometry = word['geometry']
                    word_bbox = f'bbox {word_geometry[0][0]} {word_geometry[0][1]} {word_geometry[1][0]} {word_geometry[1][1]} x_wconf {word["confidence"]:.2f}'
                    word_text = f'{word["value"]}'
                    word_hocr = f'<span class="ocrx_word" title="{word_bbox}">{word_text}</span>'
                    hocr_content.append(word_hocr)
                hocr_content.append('</span>')
            hocr_content.append('</p>')
        hocr_content.append('</div>')




### Handwritten OCR using Doctr
def handwritten_ocr(image_path, predictor, file_path):

    ## Converting images to DocumentFile
    print("Handwritten OCR using Doctr under progress")
    doc = DocumentFile.from_images(image_path)
    result = predictor(doc)
    
    print("Saving the output into text file and hocr file")
    final_values = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            values = ''
            for word in line.words:
                values += word.value + ' '
            final_values.append(values)
        final_values.append('\n')

    with open(file_path, 'w') as file:
        for line in final_values:
            file.write(line)

    hocr = get_hocr(result.export())
    soup = BeautifulSoup(hocr, 'html.parser')
    prettified_hocr = soup.prettify()

    with open(file_path[:-3] + 'hocr', "w", encoding="utf-8") as output_file:
        output_file.write(prettified_hocr)


### DocTR OCR
def detection(image_paths):
    model = ocr_predictor(pretrained=True)
    document = DocumentFile.from_images(image_paths)
    result = model(document)
    json_response = result.export()
    return json_response


def get_doctr_objs(result,file):
  json_op=detection(file)
  for page in json_op['pages']:
    y,x=page['dimensions']
    for block in page['blocks']:
      ((xl,yl),(xh,yh))=block['geometry']
      xl,yl,xh,yh=[int(xl*x),int(yl*y),int(xh*x),int(yh*y)]
      # print(xl,yl,xh,yh)
      if not any(has_overlap([xl,yl,xh,yh],b) for b in result):
        typ='Text'
        content=[]
        prev=0
        temp=''
        for line in block['lines']:
          yt=int(line['geometry'][0][1]*y)
          if abs(yt-prev)>7:
            if len(temp)>2:
              content.append(temp)
            temp=''
          prev=yt
          for word in line['words']:
            temp+=word['value']+' '
        content.append(temp)
        result.append([xl,yl,xh,yh,typ,yh-yl,content])
  def basis(x):
    return x[1]
  result=sorted(result,key=basis)
  return result,y,x


def pdf_to_txt(orig_pdf_path, project_folder_name, language_model, ocr_only, is_handwritten=False):

    output_directory = os.path.join(OUTPUT_DIR, project_folder_name)
    images_folder = os.path.join(output_directory, "Images")
    print('Output Directory is :', output_directory)

    print('Creating Directories for storing OCR outputs and data')
    for directory in DIRECTORIES:
        final_directory = os.path.join(output_directory, directory)
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
            if directory != 'Images':
                os.mknod(final_directory+'/README.md',mode=0o666)

    
    os.system('cp ' + os.path.join(RESOURCES_DIR, 'project.xml') + ' ' + output_directory)
    os.system('cp ' + os.path.join(RESOURCES_DIR, 'Dicts/') + '* '+ output_directory+"/Dicts/")

    # if input type is images, then copy the images to the output directory
    output_file=simple_counter_generator("page",".jpg")
    if IMAGE_CONVERT_OPTION and args.input_type == 'pdf' and len(os.listdir(images_folder))==0:
        print('Converting PDF to Images')  
        convert_from_path(orig_pdf_path ,output_folder=images_folder, dpi=DPI,fmt='jpeg',jpegopt=JPEGOPT,output_file=output_file)
        print("Images Creation Done.")
    elif args.input_type == 'images':
        shutil.copytree(args.input_file, images_folder)

    print(" *** STARTING OCR ENGINE *** ")
    print("Selected language model :", language_model)

    startTime = time.time()
    img_files = sorted(os.listdir(images_folder))
    cropped_figures_folder = os.path.join(output_directory, "Cropped_Images/")
    individual_output_dir = os.path.join(output_directory, "Inds/")
    print('Performing OCR on Images')

    if(is_handwritten):
        print('Handwritten OCR selected, Initializing Handwritten Models')
        predictor = initialize_handwritten_models(language_model)
    else:
        os.environ['IMAGESFOLDER'] = images_folder
        os.environ['OUTPUTDIRECTORY']= output_directory
        os.system('find $IMAGESFOLDER -maxdepth 1 -type f > $OUTPUTDIRECTORY/tmp.list')

    depth = 0
    for img_file in img_files:
        # tags = ""
        img_path = os.path.join(images_folder, img_file)
        # image = Image.open(img_path)
        # gray_image = image.convert('L')

        if(is_handwritten):
             handwritten_ocr(img_path, predictor, individual_output_dir + img_file[:-3] + 'txt')
        else:
            result = get_lpmodel_objs(img_path, [])
            result, y, x = get_doctr_objs(result, img_path)
            img = cv2.imread(img_path)
            tags = ""
            temp = 0
            for index, l in enumerate(result):
                # print(l)
                div = f"\t\t\t<div style='position:absolute;width:{str(l[2]-l[0])}px;top: {str(depth+l[1])}px;left: {str(l[0])}px;'>"
                if l[4] == "Text":
                    for i in l[6]:
                        p = f"<p style='font_size=0.5em;'>{i}</p>"
                        div += p
                elif l[4] == "Image" or l[4] == "Table":
                    crp = img[l[1] : l[3], l[0] : l[2]]
                    if not cv2.imwrite(f"{cropped_figures_folder}figure_{index}.jpg", crp):
                        raise Exception(f"Could not write image at {cropped_figures_folder}")
                    i = f'<img src="../Cropped_Images/figure_{index}.jpg"> '
                    div += i
                div += "</div>\n"
                tags += div
                # print(div)
                depth += y
            hocr = f"""
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN""http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
    <html lang="en" xml:lang="en" xmlns="http://www.w3.org/1999/xhtml">
      <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta content="ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf" name="ocr-capabilities"/>
        <title>{orig_pdf_path.split('/')[-1].split('.')[0]}</title>
      </head>
      <body>
        {tags}
      </body>
    </html>"""
            soup = BeautifulSoup(hocr, "html.parser")

            # TextAttributes Code
            # ta = TextAttributes([img_path], ocr_engine='tesseract')
            # result = ta.generate(hocr,output_type='hocr')

            # Write final hocrs
            hocrfile = individual_output_dir + img_file[:-3] + 'hocr'
            f = open(hocrfile, "w+")
            f.write(str(soup))
            # txt, hocr = printed_ocr(gray_image, language_model) Give tesseract ocr the image

            # with open(individual_output_dir +img_file[:-3] + 'txt', 'w') as f:
                # f.write(txt)

            # with open(individual_output_dir + img_file[:-3] + 'hocr', 'w+b') as f:
                # f.write(hocr)
    
    
    endTIme = time.time()
    ocr_duration = round(endTIme - startTime, 2)
    print('Done with OCR Engine of ' + str(output_file) + ' of ' + str(len(img_files)) + ' pages in ' + str(ocr_duration) + ' seconds')

    if ocr_only:
        print('OCR Only Mode Selected. Moving to Corrector output')
        corrector_directory = os.path.join(output_directory, "CorrectorOutput/")
        copy_command   = 'cp {}*.hocr {}'.format(individual_output_dir, corrector_directory)
        rename_command = "rename 's/\.hocr$/.html/' {}*.hocr".format(corrector_directory)
        os.system(copy_command)
        os.system(rename_command)
        print('Corrector Output Generated')

    print('OCR Engine Completed Successfully')

def parse_args():
    parser = argparse.ArgumentParser(description="Documents OCR Input Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--orig_pdf_path", type=str, default=None, help="path to the input pdf file")
    parser.add_argument("-it", "--input_type", type=str, default="pdf", help="type of input file | pdf/images")
    parser.add_argument("-o", "--project_folder_name", type=str, default="output_set", help="Name of the output folder")
    parser.add_argument("-l", "--language_model", type=str, default="Devangari", help="language to be used for OCR")
    parser.add_argument("-t", "--ocr_method", dest="ocr_method", action="store_true", help="OCR method : Printed/Handwritten, True if Handwritten")
    parser.add_argument("-c", "--ocr_only", dest="ocr_only", action="store_true", help="OCR only mode, stores True/False")
    
    # Not needed for now
    parser.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="save the output files")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    pdf_to_txt(args.orig_pdf_path, args.project_folder_name, args.language_model, args.ocr_only, args.ocr_method)