import os
import sys
import time

# import cv2
import shutil
import argparse
import pytesseract

from bs4 import BeautifulSoup

from pdf2image import convert_from_path


from src.config import *


# for simpler filename generation
def simple_counter_generator(prefix="", suffix=""):
    while True:
        yield 'p'


def pdf_to_txt(args):

    outputDirectory = os.path.join(OUTPUT_DIR, args.outputsetname)
    imagesFolder    = os.path.join(outputDirectory,'Images')
    print('Output Directory is :', outputDirectory)

    print("Creating Directories for storing OCR data")
    for directory in DIRECTORIES:
        final_dir = os.path.join(outputDirectory, directory)
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            if not 'Images' in final_dir:
                with open(final_dir + '/README.md', 'w') as f:
                    pass

    if not os.path.exists(outputDirectory + '/Dicts'):
        shutil.copytree(TOOLS_DIR, outputDirectory, dirs_exist_ok=True)
        # shutil.copytree(TOOLS_DIR + 'Dicts', outputDirectory + '/Dicts') 
        # shutil.copy2(TOOLS_DIR + 'project.xml', outputDirectory + '/project.xml')

    # if input type is images, then copy the images to the output directory
    if args.input_type == 'pdf' and len(os.listdir(imagesFolder))==0:
        print("Converting PDF to Images")
        output_file = simple_counter_generator("page", ".jpg")
        convert_from_path(args.input_file, output_folder=imagesFolder, dpi=300, fmt='jpeg', jpegopt= JPG_OPTIONS, output_file=output_file)
        print("Images Creation Done",end='\n\n')

    elif args.input_type == 'images':
        shutil.copytree(args.input_file, imagesFolder)


    os.environ['IMAGESFOLDER'] = imagesFolder
    os.environ['OUTPUTDIRECTORY'] = outputDirectory
    os.environ['LANGUAGE'] = args.language
    command = "find $IMAGESFOLDER -maxdepth 1 -type f > $OUTPUTDIRECTORY/tmp.list"
    os.system(command)

    languages = pytesseract.get_languages(config=TESSDATA_DIR_CONFIG)
    print(languages)



    print("***** Starting OCR Engine *****")
    startTime = time.time()
    sorted_img_files = sorted(os.listdir(imagesFolder))

    for imgfile in os.listdir(imagesFolder):

        img_path = os.path.join(imagesFolder, imgfile)

        txt_file = outputDirectory + '/Inds/' + imgfile[:-3] + 'txt'
        hocr_file = outputDirectory + '/Inds/' + imgfile[:-3] + 'hocr'


        txt = pytesseract.image_to_string(img_path, lang=args.language)
        with open(txt_file, 'w') as f:
            f.write(txt)

        print('We will OCR the image :' + imgfile)
        hocr = pytesseract.image_to_pdf_or_hocr(img_path, lang=args.language, extension='hocr')
        soup = BeautifulSoup(hocr, 'html.parser')

        # Write final hocrs
        
        with open(hocr_file, 'w+') as f:
            f.write(str(soup))

        # Generate HTMLS in Corrector Output if OCR ONLY
        if(args.ocr_only):
            corrector_output_dir = outputDirectory + "/CorrectorOutput"
            shutil.copytree(outputDirectory + '/Inds', corrector_output_dir, dirs_exist_ok=True)
            for hocrfile in os.listdir(corrector_output_dir):
                if "hocr" in hocrfile:
                    htmlfile = hocrfile.replace(".hocr", ".html")
                    os.rename(corrector_output_dir + '/' + hocrfile, corrector_output_dir + '/' + htmlfile)


    # Calculate the time elapsed for entire OCR process
    endTime = time.time()
    ocr_duration = round((endTime - startTime), 3)
    print('Done with OCR of ' + str(args.outputsetname) + ' of ' + str(len(os.listdir(imagesFolder))) + ' pages in ' + str(ocr_duration) + ' seconds')

    
    # if(args.preprocess == 'text_attributes'):
    #     print("Preprocessing images for Text Attributes")
    #     text_attributes(imagesFolder, outputDirectory)



def parse_args():
    parser = argparse.ArgumentParser(description="Documents OCR Input Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_file", type=str, default=None, help="path to the input pdf file")
    parser.add_argument("-it", "--input_type", type=str, default="pdf", help="type of input file | pdf/images")
    parser.add_argument("-o", "--outputsetname", type=str, default="output_set", help="Name of the output folder")
    parser.add_argument("-l", "--language", type=str, default="Devangari", help="language to be used for OCR")
    parser.add_argument("-t", "--ocr_type", type=str, default="printed", help="type of document to be processed | printed/handwritten/scene-text")
    parser.add_argument("-c", "--ocr_only", dest="ocr_only", action="store_true", help="OCR only mode, stores True/False")
    parser.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
    parser.add_argument("-s", "--save", dest="save", action="store_true", help="save the output files")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="debug mode")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    pdf_to_txt(args)