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
import cv2

import shutil
import argparse

import torch
import pytesseract

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



def doctr_get_hocr(json_data):

    hocr_content = []
    for page in json_data['pages']:
        y,x=page['dimensions']
        hocr_content.append(f'\t<div class="ocr_page">\n')
        for block in page['blocks']:
            block_geometry = block['geometry']
            hocr_content.append(f'\t\t<span class="ocr_carea" title="bbox {str(int(block_geometry[0][0]*x))} {str(int(block_geometry[0][1]*y))} {str(int(block_geometry[1][0]*x))} {str(int(block_geometry[1][1]*y))}">\n')
            for line in block['lines']:
                line_geometry = line['geometry']
                hocr_content.append(f'\t\t\t<span class="ocr_line" title="bbox {str(int(line_geometry[0][0]*x))} {str(int(line_geometry[0][1]*y))} {str(int(line_geometry[1][0]*x))} {str(int(line_geometry[1][1]*y))}" style="position:absolute;top: {str(int(line_geometry[0][1]*y))}px;left: {str(int(line_geometry[0][0]*x))}px;">\n')
                for word in line['words']:
                    word_geometry = word['geometry']
                    word_bbox = f'bbox {str(int(word_geometry[0][0]*x))} {str(int(word_geometry[0][1]*y))} {str(int(word_geometry[1][0]*x))} {str(int(word_geometry[1][1]*y))} x_wconf {word["confidence"]:.2f}'
                    word_text = f'{word["value"]}'
                    word_hocr = f'\t\t\t\t<span class="ocrx_word" title="{word_bbox}" style="display:inline-block">{word_text}</span>\n'
                    hocr_content.append(word_hocr)
                hocr_content.append('\t\t\t</span>\n')
            hocr_content.append('\t\t</span>\n')
        hocr_content.append('\t</div>\n')

    # Combine content and create the complete HOCR document
    complete_hocr = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Output</title>
</head>
<body>
{" ".join(hocr_content)}
</body>
</html>'''
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

    hocr = doctr_get_hocr(result.export())
    return hocr

# to get the tesseract output's objects
def get_tesseract_objs(img,language,cf):
  tesseract_output=pytesseract.image_to_data(img,lang=language,config=cf).split('\n')
  words=[]
  for row in tesseract_output:
    row=row.split()
    try:
      if row[11]!='-1':
        words.append(row)
    except:
      pass
  line_dict={}
  objs=[]
  for idx,word in enumerate(words):
    if idx!=0:
      block=word[2]
      para=word[3]
      line=word[4]
      text=word[11]
      x=int(word[6])
      y=int(word[7])
      w=int(word[8])
      h=int(word[9])
      if block not in line_dict.keys():
        line_dict[block]={}
        line_dict[block][para]={}
        line_dict[block][para][line]=[x,y,w,h,[[x,y,x+w,y+h,text]]]
      else:
        if para not in line_dict[block].keys():
          line_dict[block][para]={}
          line_dict[block][para][line]=[x,y,w,h,[[x,y,x+w,y+h,text]]]
        else:
          if line not in line_dict[block][para].keys():
            line_dict[block][para][line]=[x,y,w,h,[[x,y,x+w,y+h,text]]]
          else:
            line_dict[block][para][line][0]=min(line_dict[block][para][line][0],x)
            line_dict[block][para][line][1]=min(line_dict[block][para][line][1],y)
            line_dict[block][para][line][2]=x+w
            line_dict[block][para][line][3]=max(line_dict[block][para][line][3],h)
            line_dict[block][para][line][4].append([x,y,x+w,y+h,text])
  for block in line_dict.keys():
    for para in line_dict[block].keys():
      for line in line_dict[block][para].keys():
        word=line_dict[block][para][line]
        objs.append([word[0],word[1],word[2]+word[0],word[3]+word[1],'Text',word[4]])
  objs=sorted(objs,key=lambda x:x[1])
  return objs

def tesseract_get_hocr(img,language,cf):
    body=''

    #  getting the tesseract objects
    result=get_tesseract_objs(img,language,cf)

    # making the hocr output
    # bbox = [ xmin , ymin , xmax , ymax , type , list of words]
    for index,bbox in enumerate(result):
        span=f"\t<span class='ocr_line' title='bbox {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}' style='position:absolute;top: {str(bbox[1])}px;left: {str(bbox[0])}px;'>\n"
        for word in bbox[5]:
            span+=f"\t\t<span class='ocr_word' title='bbox {word[0]} {word[1]} {word[2]} {word[3]}' style='display:inline-block' >"+word[4]+'</span>\n'
        span+='\t</span>\n'
        body+=span
    # making the html file
    html=f'''
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Output</title>
</head>
<body>
{body}
</body>
</html>'''
    # return the html code
    return html


### Printed OCR using Tesseract
def printed_ocr(gray_image, language_model):
    txt = pytesseract.image_to_string(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
    # hocr = pytesseract.image_to_pdf_or_hocr(gray_image, lang=language_model, extension='hocr', config=TESSDATA_DIR_CONFIG)
    hocr= tesseract_get_hocr(gray_image,language_model,TESSDATA_DIR_CONFIG)
    return txt, hocr




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

        #  getting the tables
        print('Performing Table Detection',end=' ')
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

        image = Image.open(img_path)
        gray_image = image.convert('L')

        if(is_handwritten):
            hocr=handwritten_ocr(img_path, predictor, individual_output_dir + img_file[:-3] + 'txt')
        else:
            txt, hocr = printed_ocr(gray_image, language_model)

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

    print('OCR Engine Completed Successfully , The Text Outputs Are In /Inds and Hocr Outputs IN /ProcessedImages')

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
# pdf_to_txt('BCC_BR_115_434.pdf','output_set','eng',False)