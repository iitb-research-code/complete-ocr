# How to run
# python3 pycodes/pdf_to_txt_tesseract_ocr.py test.pdf OUTPUTSETNAME language ocr_only
import sys
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import os
from pdf2image import convert_from_path
import cv2
from bs4 import BeautifulSoup
import time
import sys
from text_attributes import TextAttributes

output_dir = './output_books/'

def parse_boolean(b):
    return b == "True"

# for simpler filename generation
def simple_counter_generator(prefix="", suffix=""):
    i = 400
    while True:
        i += 1
        yield 'p'

def pdf_to_txt(orig_pdf_path, project_folder_name, lang, ocr_only, pdftoimg):
    outputDirIn = output_dir
    outputDirectory = outputDirIn + project_folder_name
    print('output directory is ', outputDirectory)
    # create images,text folder
    print('cwd is ', os.getcwd())
    if not os.path.exists(outputDirectory):
        os.mkdir(outputDirectory)

    if not os.path.exists(outputDirectory + "/Images"):
        os.mkdir(outputDirectory + "/Images")

    imagesFolder = outputDirectory + "/Images"
    imageConvertOption = 'True'

    print("converting pdf to images")
    jpegopt = {
        "quality": 100,
        "progressive": True,
        "optimize": False
    }

    output_file = simple_counter_generator("page", ".jpg")
    print('orig pdf oath is', orig_pdf_path)
    print('cwd is', os.getcwd())
    print("orig_pdf_path is", orig_pdf_path)
    if (parse_boolean(imageConvertOption)):
        convert_from_path(orig_pdf_path, output_folder=imagesFolder, dpi=300, fmt='jpeg', jpegopt=jpegopt,
                          output_file=output_file)

    print("images created.")
    print(pdftoimg)
    if pdftoimg != 'pdf2img':
        print("Now we will OCR")
        os.environ['IMAGESFOLDER'] = imagesFolder
        os.environ['OUTPUTDIRECTORY'] = outputDirectory
        tessdata_dir_config = r'--psm 3 --tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata/"'
        languages = pytesseract.get_languages(config=tessdata_dir_config)
        lcount = 0
        tesslanglist = {}
        for l in languages:
            if not (l == 'osd'):
                tesslanglist[lcount] = l
                lcount += 1
                print(str(lcount) + '. ' + l)

    print("Selected language model " + lang)

    os.environ['CHOSENMODEL'] = lang  # tesslanglist[int(linput)-1]
    if not os.path.exists(outputDirectory + "/CorrectorOutput"):
        os.mkdir(outputDirectory + "/CorrectorOutput")
        os.mknod(outputDirectory + "/CorrectorOutput/" + 'README.md', mode=0o666)

    # Creating Final set folders and files
    if not os.path.exists(outputDirectory + "/Comments"):
        os.mkdir(outputDirectory + "/Comments")
        os.mknod(outputDirectory + "/Comments/" + 'README.md', mode=0o666)
    if not os.path.exists(outputDirectory + "/VerifierOutput"):
        os.mkdir(outputDirectory + "/VerifierOutput")
        os.mknod(outputDirectory + "/VerifierOutput/" + 'README.md', mode=0o666)

    if not os.path.exists(outputDirectory + "/Inds"):
        os.mkdir(outputDirectory + "/Inds")
        os.mknod(outputDirectory + "/Inds/" + 'README.md', mode=0o666)
    if not os.path.exists(outputDirectory + "/Dicts"):
        os.mkdir(outputDirectory + "/Dicts")
        os.mknod(outputDirectory + "/Dicts/" + 'README.md', mode=0o666)
    if not os.path.exists(outputDirectory + "/Cropped_Images"):
        os.mkdir(outputDirectory + "/Cropped_Images")
    if not os.path.exists(outputDirectory + "/MaskedImages"):
        os.mkdir(outputDirectory + "/MaskedImages")

    individualOutputDir = outputDirectory + "/Inds"

    startOCR = time.time()

    if pdftoimg == 'pdf2img':
        exit(0)

    for imfile in os.listdir(imagesFolder):
        image_path = imagesFolder + "/" + imfile

        dash = imfile.index('-')
        dot = imfile.index('.')
        page = int(imfile[dash + 1 : dot])
        
        # Now we detect text using Tesseract
        print('We will OCR the image ' + image_path)
        hocr = pytesseract.image_to_pdf_or_hocr(image_path, lang=lang, extension='hocr')

        ta = TextAttributes([image_path], ocr_engine='tesseract')
        result = ta.generate(hocr,output_type='hocr')

        # Write final hocrs
        hocrfile = individualOutputDir + '/' + imfile[:-3] + 'hocr'
        f = open(hocrfile, 'w+')
        f.write(str(result))


    # Generate HTMLS in Corrector Output if OCR ONLY
    ocr_only = True
    if(ocr_only):
        copy_command = 'cp {}/*.hocr {}/'.format(individualOutputDir, outputDirectory + "/CorrectorOutput")
        os.system(copy_command)
        correctorFolder = outputDirectory + "/CorrectorOutput"
        for hocrfile in os.listdir(correctorFolder):
            if "hocr" in hocrfile:
                htmlfile = hocrfile.replace(".hocr", ".html")
                os.rename(correctorFolder + '/' + hocrfile, correctorFolder + '/' + htmlfile)

    
    # Calculate the time elapsed for entire OCR process
    endOCR = time.time()
    ocr_duration = round((endOCR - startOCR), 3)
    print('Done with OCR of ' + str(project_folder_name) + ' of ' + str(len(os.listdir(imagesFolder))) + ' pages in ' + str(ocr_duration) + ' seconds')
    #return is_machineReadable
    return False

# Function Calls
input_file= sys.argv[1]
outputsetname = sys.argv[2]
lang = sys.argv[3]
ocr_only = sys.argv[4]
pdf_to_txt(input_file, outputsetname, lang, ocr_only, '')