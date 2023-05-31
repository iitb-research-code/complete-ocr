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
import layoutparser as lp
import cv2
from bs4 import BeautifulSoup
import time
import sys
from table_cellwise_detection import get_final_table_hocrs_from_image
from ocr_config import output_dir

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
        # os.environ['CWD']='/home/sanskar/udaan-deploy-pipeline'
        os.environ['OUTPUTDIRECTORY'] = outputDirectory
        # os.environ['CHOSENFILENAMEWITHNOEXT']=chosenFileNameWithNoExt
        # os.system('find $IMAGESFOLDER -maxdepth 1 -type f > $OUTPUTDIRECTORY/tmp.list')
        # tessdata_dir_config = r'--tessdata-dir "$/home/sanskar/NLP-Deployment-Heroku/udaan-deploy-pipeline/tesseract-exec/tessdata/"'
        tessdata_dir_config = r'--psm 3 --tessdata-dir "/home/ayush/udaan-deploy-flask/udaan-deploy-pipeline/tesseract-exec/share/tessdata/"'
        tessdata_dir_config = r'--psm 3 --tessdata-dir "/usr/share/tesseract-ocr/4.00/tessdata/"'
        tessdata_dir_config = r'--tessdata-dir "/home/ayush/udaan-deploy-flask/udaan-deploy-pipeline/tesseract-exec/share/tessdata/"'
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

    os.system('cp ../../project.xml ' + outputDirectory)
    os.system('cp ../../dicts/* ' + outputDirectory + "/Dicts/")
    individualOutputDir = outputDirectory + "/Inds"

    startOCR = time.time()

    if pdftoimg == 'pdf2img':
        exit(0)

    ### Layout Parser Model Configuration
    model_config = 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config'
    extra_config = "MODEL.ROI_HEADS.SCORE_THRESH_TEST"
    label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    model = lp.Detectron2LayoutModel(model_config, extra_config=[extra_config, 0.8], label_map=label_map)
    # layout = model.detect(image)

    for imfile in os.listdir(imagesFolder):
        finalimgtoocr = imagesFolder + "/" + imfile

        dash = imfile.index('-')
        dot = imfile.index('.')
        page = int(imfile[dash + 1 : dot])
        
        # Get tables from faster rcnn predictions in hocr format
        fullpathimgfile = imagesFolder + '/' + imfile
        tabledata = get_tables_from_page(fullpathimgfile)
        # tabledata = []

        # Write txt files for all pages using Tesseract
        txt = pytesseract.image_to_string(imagesFolder + "/" + imfile, lang=lang)
        with open(individualOutputDir + '/' + imfile[:-3] + 'txt', 'w') as f:
            f.write(txt)

        # Hide all tables from images before perfroming recognizing text 
        if len(tabledata) > 0:
            img = cv2.imread(imagesFolder + "/" + imfile)
            for entry in tabledata:
                bbox = entry[1]
                tab_x = bbox[0]
                tab_y = bbox[1]
                tab_x2 = bbox[2]
                tab_y2 = bbox[3]
                img_x = int(tab_x)
                img_y = int(tab_y)
                img_x2 = int(tab_x2)
                img_y2 = int(tab_y2)
                cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (255, 0, 255), -1)
            finalimgfile = outputDirectory + "/MaskedImages/" + imfile[:-4] + '_filtered.jpg'
            cv2.imwrite(finalimgfile, img)
            finalimgtoocr = finalimgfile

        # Perform figure detection from page image to get their hocrs and bounding boxes
        img = cv2.imread(imagesFolder + "/" + imfile)
        storeMaskedImages = False
        figuredata = get_images_from_page_image(model, img, outputDirectory, imfile, page, storeMaskedImages)
        #figuredata = []
        if len(figuredata) > 0 and storeMaskedImages:
            # Masked Image will be sent to tesseract
            finalimgtoocr = outputDirectory + "/MaskedImages/" + imfile[:-4] + '_filtered.jpg'
        
        # Now we detect text using Tesseract
        print('We will OCR the image ' + finalimgtoocr)
        hocr = pytesseract.image_to_pdf_or_hocr(finalimgtoocr, lang=lang, extension='hocr')
        soup = BeautifulSoup(hocr, 'html.parser')
        
        # Adding table hocr in final hocr at proper position
        if len(tabledata) > 0:
            for entry in tabledata:
                tab_tag = entry[0]
                tab_element = BeautifulSoup(tab_tag, 'html.parser')
                # print(tab_tag)
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

        # Write final hocrs
        hocrfile = individualOutputDir + '/' + imfile[:-3] + 'hocr'
        f = open(hocrfile, 'w+')
        f.write(str(soup))


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


def get_tables_from_page(fullpathimgfile):
    # Return list of table HOCR and bbox here
    result = get_final_table_hocrs_from_image(fullpathimgfile)
    print(str(fullpathimgfile) + ' has ' + str(len(result)) + ' tables extracted')
    return result


def get_images_from_page_image(model, image, outputDirectory, imfile, pagenumber, storeMasked):
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

    # Storing masked output:
    if figure_count > 0 and storeMasked:
        finalimgfile = outputDirectory + "/MaskedImages/" + imfile[:-4] + '_filtered.jpg'
        print(finalimgfile)
        cv2.imwrite(finalimgfile, image)
    return result

# Function Calls
input_file= sys.argv[1]
outputsetname = sys.argv[2]
lang = sys.argv[3]
ocr_only = sys.argv[4]
pdf_to_txt(input_file, outputsetname, lang, ocr_only, '')

