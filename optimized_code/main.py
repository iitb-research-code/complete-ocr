import os
import cv2
import time
import shutil
import argparse
import warnings
import pytesseract
import layoutparser as lp

from pdf2image import convert_from_path
from bs4 import BeautifulSoup
from tqdm import tqdm


# from table_cellwise_detection import get_final_table_hocrs_from_image
from src.config import *
from src.utils import get_final_table_hocrs_from_image

warnings.filterwarnings('ignore')


# for simpler filename generation
def simple_counter_generator(prefix="", suffix=""):
    while True:
        yield 'p'
# to get the tesseract output's objects
def get_tesseract_objs(img_path,language):
  img=cv2.imread(img_path)
  tesseract_output=pytesseract.image_to_data(img,lang=language,config='--psm 6').split('\n')
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
        print([word[0],word[1],word[2]+word[0],word[3]+word[1],'Text',word[4]])
  objs=sorted(objs,key=lambda x:x[1])
  return objs


# to generate the hocr output from the tesseract outputs
def hocr_output(file,language):
    body=''

    #  getting the tesseract objects
    result=get_tesseract_objs(file,language)

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
  <title>{file.split('/')[-1].split('.')[0]}</title>
</head>
<body>
{body}
</body>
</html>'''
    # return the html code
    return html

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
            imagehocr = f"<img class=\"ocr_im\" title=\"bbox {x1} {y1} {x2} {y2}\" style='position:absolute;top: {str(y1)}px;left: {str(x1)}px;' src=\"../{image_file_name}\">"
            result.append([imagehocr, bbox])

    return result


def pdf_to_txt(args):

    outputDirectory = os.path.join(OUTPUT_DIR, args.outputsetname)
    imagesFolder    = os.path.join(outputDirectory,'Images')
    print('Output Directory is :', outputDirectory)

    print("Creating Directories for storing OCR data")
    for directory in directories:
        final_dir = os.path.join(outputDirectory, directory)
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
            if not 'Images' in final_dir:
                with open(final_dir + '/README.md', 'w') as f:
                    pass


    print("Converting PDF to Images")


    if len(os.listdir(imagesFolder))==0:
        output_file = simple_counter_generator("page", ".jpg")
        convert_from_path(args.input_file, output_folder=imagesFolder, dpi=300, fmt='jpeg', jpegopt= jpg_options, output_file=output_file)

    print("Images Creation Done",end='\n\n')


    model = lp.Detectron2LayoutModel(model_config, extra_config=[extra_config, 0.8], label_map=label_map)


    print("***** Starting OCR Engine *****")
    startTime = time.time()
    sorted_img_files = sorted(os.listdir(imagesFolder))

    for imgfile in tqdm(sorted_img_files):

        img_path = os.path.join(imagesFolder, imgfile)
        img = cv2.imread(img_path)

        dash, dot = imgfile.index('-'), imgfile.index('.')
        page = int(imgfile[dash + 1 : dot])
        

        txt = pytesseract.image_to_string(img_path, lang=args.language)
        with open(outputDirectory + '/Inds/' + imgfile[:-3] + 'txt', 'w') as f:
            f.write(txt)

        
        tabledata = get_final_table_hocrs_from_image(img_path)
        # Hide all tables from images before perfroming recognizing text 
        if len(tabledata) > 0:
            for entry in tabledata:
                bbox = entry[1]
                x1, y1, x2, y2 = list(map(int,bbox))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), -1)
            img_path = outputDirectory + "/MaskedImages/" + imgfile[:-4] + '_filtered.jpg'
            cv2.imwrite(img_path, img)

        # Perform figure detection from page image to get their hocrs and bounding boxes
        figuredata = get_images_from_page_image(model, img, outputDirectory, page)

            # Storing masked output:
        if len(figuredata) > 0 and storeMaskedImages:
            img_path = outputDirectory + "/MaskedImages/" + imgfile[:-4] + '_filtered.jpg'
            cv2.imwrite(img_path, img)

        

        print('We will OCR the image :' + imgfile)
        hocr = hocr_output(img_path, lang=args.language)
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

        # Write final hocrs
        hocrfile = outputDirectory + '/Inds/' + imgfile[:-3] + 'hocr'
        f = open(hocrfile, 'w+')
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
        
        
def parse_args():
    parser = argparse.ArgumentParser(description="PDF Documents Input", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_file", type=str, default=None, help="path to train data folder(s)")
    parser.add_argument("-o", "--outputsetname", type=str, default=None, help="path to val data folder")
    parser.add_argument("-l", "--language", type=str, default=None, help="path to val data folder")
    parser.add_argument("--ocr_only", dest="ocr_only", action="store_true", help="Run the validation loop")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    pdf_to_txt(args)

