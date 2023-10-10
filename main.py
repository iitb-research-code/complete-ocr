'''
THe script is used to convert pdf to txt using Doctr and Tesseract OCR
It is the combination of both handwritten and printed OCR using Doctr and Tesseract respectively
The script is called from the main script in the pipeline
It contains entire OCR pipeline engine end to end

Author: Badri Vishal Kasuba, Pranav Dalvi
'''


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

    # Combine content and create the complete HOCR document
    complete_hocr = f'<!DOCTYPE html><html><head><title></title></head><body>{" ".join(hocr_content)}</body></html>'
    return complete_hocr



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




### Printed OCR using Tesseract
# def printed_ocr(gray_image, language_model):
#     txt = pytesseract.image_to_string(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
#     hocr = pytesseract.image_to_pdf_or_hocr(gray_image, lang=language_model, extension='hocr', config=TESSDATA_DIR_CONFIG)
#     return txt, hocr


# TESSERACT DETECTED TEXT OBJECTS (Text OCR)
def get_tesseract_objs(result1, img_path, lang):
    img = cv2.imread(img_path)
    words = pytesseract.image_to_data(img, lang=lang, config="--psm 6").split("\n")
    l = []
    for i in words:
        i = i.split()
        try:
            if i[11] != "-1":
                # print(i)
                l.append(i)
        except:
            pass
    d = {}
    for idx, text in enumerate(l):
        if idx != 0:
            block = text[2]
            para = text[3]
            line = text[4]
            data = text[11]
            x = int(text[6])
            y = int(text[7])
            w = int(text[8])
            h = int(text[9])
            conf = (text[10])
            if block not in d.keys():
                d[block] = {}
                d[block][para] = {}
                d[block][para][line] = [x, y, w, h, data, block, para, line]
            else:
                if para not in d[block].keys():
                    d[block][para] = {}
                    d[block][para][line] = [x, y, w, h, data, block, para, line]
                else:
                    if line not in d[block][para].keys():
                        d[block][para][line] = [x, y, w, h, data, block, para, line]
                    else:
                        d[block][para][line][0] = min(d[block][para][line][0], x)
                        d[block][para][line][1] = min(d[block][para][line][1], y)
                        d[block][para][line][2] += w
                        d[block][para][line][3] = max(d[block][para][line][3], h)
                        d[block][para][line][4] += " " + data
                        d[block][para][line][5] = block
                        d[block][para][line][6] = para
                        d[block][para][line][7] = line
    for block in d.keys():
        for para in d[block].keys():
            for line in d[block][para].keys():
                word = d[block][para][line]
                print([word[0], word[1], word[2] + word[0], word[3] + word[1]])
                if not any(has_overlap([word[0], word[1], word[2] + word[0], word[3] + word[1]], b)for b in result1):
                    result1.append([word[0],word[1],word[2] + word[0],word[3] + word[1],"Text",word[3],word[4],word[5],word[6],word[7]])
    result1 = sorted(result1, key=lambda x: x[1])
    return result1

def generate_hocr(result1, img_path, language_model):

    hocr = pytesseract.image_to_pdf_or_hocr(img_path, extension="hocr", lang=language_model)

    # TextAttributes Code
    ta = TextAttributes([img_path], ocr_engine='tesseract')
    result = ta.generate(hocr,output_type='hocr')

    soup = BeautifulSoup(result, 'html.parser')
    line_tags = soup.find_all('span',class_="ocr_line")

    # Extract Required Data from hOCR
    hocr_info_list = []
    for line_tag in line_tags:
        bbox_value = line_tag['title'].split(';')[0].split(' ')[1:]
        bbox_value_int = [int(val) for val in bbox_value]

        # Check if any of the ocrx_word tags have text content to avoid overlap by empty span tags (ocrx_word)
        word_tags = line_tag.find_all('span', class_="ocrx_word")
        has_text = any(word_tag.get_text(strip=True) for word_tag in word_tags)

        if has_text:
            hocr_info_list.append([bbox_value_int, word_tags])

    for index, hocr_info in enumerate(hocr_info_list):
        if not any(has_overlap([hocr_info[0][0], hocr_info[0][1], hocr_info[0][2], hocr_info[0][3]], b)for b in result1):
            result1.append([hocr_info[0][0],hocr_info[0][1],hocr_info[0][2],hocr_info[0][3],"Text",hocr_info[0][3],hocr_info[1]])
    result1 = sorted(result1, key=lambda x: x[1])

    return result1



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

        img_path = os.path.join(images_folder, img_file)
        # image = Image.open(img_path)
        # gray_image = image.convert('L')

        if(is_handwritten):
             handwritten_ocr(img_path, predictor, individual_output_dir + img_file[:-3] + 'txt')
        else:
            result = get_lpmodel_objs(img_path, [])
            result = generate_hocr(result, img_path, language_model)
            # result = get_tesseract_objs(result, img_path, language_model)
            img = cv2.imread(img_path)
            tags = ""
            temp = 0
            depth = 0
            for index, l in enumerate(result):
                div = f"\t\t\t<div class='ocr_carea' id='block' style='position:absolute;width:{str(l[2]-l[0])}px;top: {str(depth+l[1])}px;left: {str(l[0])}px;'>\n"
                if l[4] == "Text":
                    p = f"<p class='ocr_par' id='par'>\n"
                    span_line = f"<span class='ocr_line' id='line' title='bbox {str(l[0])} {str(l[1])} {str(l[2])} {str(l[3])};' >\n"
                    for index, span in enumerate(l[6]):
                        span = f"{span}\n"
                        span_line += span
                    span_line += "</span>\n"
                    p += span_line
                    p += "</p>\n"
                    div += p
                elif l[4] == "Image" or l[4] == "Table":
                    crp = img[l[1] : l[3], l[0] : l[2]]
                    if not cv2.imwrite(f"{cropped_figures_folder}figure_{index}.jpg", crp):
                        raise Exception(f"Could not write image at {cropped_figures_folder}")
                    i = f'<img src="../Cropped_Images/figure_{index}.jpg"> '
                    div += i
                div += "</div>\n"
                tags += div
                temp = l[3]
            depth += temp + 200
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

            # Write final hocrs
            hocrfile = individual_output_dir + img_file[:-3] + 'hocr'
            f = open(hocrfile, "w+")
            f.write(str(soup))
    
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