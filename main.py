'''
THe script is used to convert pdf to txt using Doctr and Tesseract OCR
It is the combination of both handwritten and printed OCR using Doctr and Tesseract respectively
The script is called from the main script in the pipeline
It contains entire OCR pipeline engine end to end

Author: Badri Vishal Kasuba
'''


# Import Libraries
import os
import time

import shutil
import argparse

import torch
import pytesseract

import cv2
import layoutparser as lp

# use torch environment
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

from pdf2image import convert_from_path
from PIL import Image
from bs4 import BeautifulSoup

from utility.config import *
from utility.text_attributes import TextAttributes
from utility.utils import get_final_table_hocrs_from_image


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

    ta = TextAttributes([image_path], 'doctr', thres=args.bold_threshold, k_size=args.kernel_size)
    hocr = ta.generate(result.export(),"hocr")

    return hocr


### Printed OCR using Tesseract
def printed_ocr(img_path,gray_image, language_model):
    txt = pytesseract.image_to_string(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
    hocr = pytesseract.image_to_pdf_or_hocr(gray_image, lang=language_model, extension='hocr', config=TESSDATA_DIR_CONFIG)
    ta = TextAttributes(images=[img_path],ocr_engine="tesseract",thres=args.bold_threshold)
    hocr = ta.generate(hocr,"hocr")
    return txt, hocr


def get_images_from_page_image(model, image, outputDirectory, pagenumber):
    layout = model.detect(image)
    result = []
    # Figure Extraction
    figure_count = 0
    for layer in layout:
        if(layer.type=='Figure'):
            x1, y1, x2, y2 = tuple(int(num) for num in layer.block.coordinates)
            cropped_image = image[y1: y2, x1: x2]
            image_file_name = '/Cropped_Images/figure_' + str(pagenumber) + '_' + str(figure_count) + '.jpg'
            cv2.imwrite(outputDirectory + image_file_name, cropped_image)
            figure_count += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), -1)
            bbox = [x1, y1, x2, y2]
            imagehocr = f"<img class=\"ocr_im\" title=\"bbox {x1} {y1} {x2} {y2}\" src=\"../{image_file_name}\">"
            result.append([imagehocr, bbox])

    return result


def pdf_to_txt(orig_pdf_path, project_folder_name, language_model, ocr_only, is_handwritten=False):

    output_directory = os.path.join(OUTPUT_DIR, project_folder_name)
    images_folder = os.path.join(output_directory, "Images")
    print('Output Directory is :', output_directory)
    print('images_folder: ',images_folder)

    print('Creating Directories for storing OCR outputs and data')
    for directory in DIRECTORIES:
        final_directory = os.path.join(output_directory, directory)
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
            if directory != 'Images':
                os.mknod(final_directory+'/README.md',mode=0o666)

    
    os.system('cp ' + os.path.join(RESOURCES_DIR, 'project.xml') + ' ' + output_directory)
    os.system('cp ' + os.path.join(RESOURCES_DIR, 'dicts/') + '* '+ output_directory+"/Dicts/")

    # if input type is images, then copy the images to the output directory
    output_file=simple_counter_generator("page",".jpg")
    if IMAGE_CONVERT_OPTION and args.input_type == 'pdf' and len(os.listdir(images_folder))==0:
        print('Converting PDF to Images')  
        convert_from_path(orig_pdf_path ,output_folder=images_folder, dpi=DPI,fmt='jpeg',jpegopt=JPEGOPT,output_file=output_file)
        print("Images Creation Done.")
    elif args.input_type == 'images':
        shutil.copytree(args.orig_pdf_path, images_folder)

    model = lp.Detectron2LayoutModel(model_config, extra_config=[extra_config, 0.8], label_map=label_map)

    print(" *** STARTING OCR ENGINE *** ")
    print("Selected language model :", language_model)

    startTime = time.time()
    img_files = sorted(os.listdir(images_folder))
    individual_output_dir = os.path.join(output_directory, "Inds/")
    ProcessedOutput=os.path.join(output_directory,'ProcessedImages/')
    print('Performing OCR on Images')

    if(is_handwritten):
        print('Handwritten OCR selected, Initializing Handwritten Models')
        predictor = initialize_handwritten_models(language_model)
    else:
        os.environ['IMAGESFOLDER'] = images_folder
        os.environ['OUTPUTDIRECTORY']= output_directory
        os.system('find $IMAGESFOLDER -maxdepth 1 -type f > $OUTPUTDIRECTORY/tmp.list')

    for img_file in img_files:

        img_path = os.path.join(images_folder, img_file)
        img=cv2.imread(img_path)

        dash, dot = img_file.index('-'), img_file.index('.')
        page = int(img_file[dash + 1 : dot])

        #  getting the tables
        print('Performing Table Detection',end=' -> ')
        tabledata = get_final_table_hocrs_from_image(img_path)
        print('Table Detection Done')

        # Hide all tables from images before perfroming recognizing text 
        if len(tabledata) > 0:
            for entry in tabledata:
                bbox = entry[1]
                x1, y1, x2, y2 = list(map(int,bbox))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), -1)
            img_path = output_directory + "/MaskedImages/" + img_file[:-4] + '_filtered.jpg'
            cv2.imwrite(img_path, img)

        # Perform figure detection from page image to get their hocrs and bounding boxes
        figuredata = get_images_from_page_image(model, img, output_directory, page)

            # Storing masked output:
        if len(figuredata) > 0 and storeMaskedImages:
            img_path = output_directory + "/MaskedImages/" + img_file[:-4] + '_filtered.jpg'
            cv2.imwrite(img_path, img)

        image = Image.open(img_path)
        gray_image = image.convert('L')

        if(is_handwritten):
            hocr=handwritten_ocr(img_path, predictor, individual_output_dir + img_file[:-3] + 'txt')
        else:
            txt, hocr = printed_ocr(img_path,gray_image, language_model)

            with open(individual_output_dir +img_file[:-3] + 'txt', 'w') as f:
                f.write(txt)

        soup = BeautifulSoup(hocr, 'html.parser')
        # Adding table hocr in final hocr at proper position
        if len(tabledata) > 0:
            for entry in tabledata:
                tab_tag = entry[0]
                tab_element = BeautifulSoup(tab_tag, 'html.parser')
                tab_bbox = entry[1]
                # y-coordinate
                tab_position = tab_bbox[1]
                for elem in soup.find_all('span', class_="ocr_line"):
                    find_all_ele = elem.attrs["title"].split(" ")
                    line_position = int(find_all_ele[2])
                    if tab_position < line_position:
                        elem.insert_before(tab_element)
                        break

        # Adding image hocr in final hocr at proper position
        if len(figuredata) > 0:
            for image_details in figuredata:
                imghocr = image_details[0]
                img_element = BeautifulSoup(imghocr, 'html.parser')
                img_position = image_details[1][1]
                for elem in soup.find_all('span', class_="ocr_line"):
                    find_all_ele = elem.attrs["title"].split(" ")
                    line_position = int(find_all_ele[2])
                    if img_position < line_position:
                        elem.insert_before(img_element)
                        break

        with open(ProcessedOutput + img_file[:-3] + 'html', 'w') as f:
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

    print('OCR Engine Completed Successfully , The Text Outputs Are In ./output_books/output_set/Inds/ and Hocr Outputs IN ./output_books/output_set/ProcessedImages/')

def parse_args():
    parser = argparse.ArgumentParser(description="Documents OCR Input Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--orig_pdf_path", type=str, default=None, help="path to the input pdf file")
    parser.add_argument("-it", "--input_type", type=str, default="pdf", help="type of input file | pdf/images")
    parser.add_argument("-o", "--project_folder_name", type=str, default="output_set", help="Name of the output folder")
    parser.add_argument("-l", "--language_model", type=str, default="Devangari", help="language to be used for OCR")
    parser.add_argument("-t", "--ocr_method", dest="ocr_method", action="store_true", help="OCR method : Printed/Handwritten, True if Handwritten")
    parser.add_argument("-c", "--ocr_only", dest="ocr_only", action="store_true", help="OCR only mode, stores True/False")
    parser.add_argument("-b", "--bold_threshold",type=float,default=0.3, help="Threshold for bold classification. Lower means more sensitive")
    parser.add_argument("-k", "--kernel_size",type=int,default=4, help="Kernel Size. Lower for smaller font. Only for Handwritten")

    
    # Not needed for now
    parser.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="save the output files")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = parse_args()
    pdf_to_txt(args.orig_pdf_path, args.project_folder_name, args.language_model, args.ocr_only, args.ocr_method)