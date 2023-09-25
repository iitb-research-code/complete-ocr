from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


class TextAttributes:
    """
    Adding Text Attributes to OCR predictions given from DocTR or Tesseract.
    Current text attributes provided:
        DocTR - Bold(Manual), Color
        Tesseract - Bold(Automatic), Color, Font Size
    """

    def __init__(self,images: list[str],ocr_engine: str, thres: float = 0.3, k_size: int = 4)-> None: 
        """
        Initializes a text attribute generator.
        
        Parameters:
            images: A list of paths of images used to perform ocr (The ocr must have been performed on these images without any type conversions)
            ocr_engine: Ocr Engine used ('doctr'/'tesseract')
            thres: Threshold for bold detection. Lower value means more more sensitive.
            k_size: Kernel size for morphological opening. Automatically choosen when using Tesseract
        """

        self.ocr_engine = ocr_engine
        self.images = [cv2.imread(img).astype("uint8") for img in images]
        self.thres = thres
        self.k_size = k_size
        self.bin_images = [self.binarize(image) for image in self.images]
        if self.ocr_engine == 'doctr':
            self.processed_images = [self.morph_open(image, self.k_size) for image in self.bin_images]
        else:
            self.processed_images = [None for image in self.bin_images]

    def generate(self,ocr_output,output_type: str)->str|dict:
        """
        Returns modified ocr_output with text attributes in required output_type

        Parameters:
            ocr_output: string (hocr) if ocr_engine is tesseract, dictionary (json) if ocr_engine is docTR
            output_type: required output_type ('hocr'/'json')
        """
        
        if type(self.processed_images[0])==type(None) and self.ocr_engine == 'tesseract':
            self.process_images_with_sizes(ocr_output)

        if self.ocr_engine == 'tesseract' and output_type == 'hocr':
            return self.parse_tesseract_to_hocr(ocr_output)

        if self.ocr_engine == 'tesseract' and output_type == 'json':
            return self.parse_tesseract_to_json(ocr_output)
    
        if self.ocr_engine == 'doctr' and output_type == 'json':
            return self.parse_doctr_to_json(ocr_output)
        
        if self.ocr_engine == 'doctr' and output_type == 'hocr':
            return self.parse_doctr_to_hocr(ocr_output)
        
    def parse_doctr_to_json(self,doc):
        """Returns json with text attributes from doctr output"""

        result_doc = json.loads(json.dumps(doc))
        for p,page in enumerate(doc["pages"]):
            for b,block in enumerate(page["blocks"]):
                for l,line in enumerate(block["lines"]):
                    for w,word in enumerate(line["words"]):

                        word_img = self.get_image(self.images[p],page["dimensions"],word["geometry"])
                        bin_reg = self.get_image(self.bin_images[p],page["dimensions"],word["geometry"])
                        processed_reg = self.get_image(self.processed_images[p],page["dimensions"],word["geometry"])

                        bold = self.isbold(processed_reg,bin_reg,self.thres)
                        color = self.get_color(word_img,mode="dark")

                        result_doc["pages"][p]["blocks"][b]["lines"][l]["words"][w]["font_weight"]="bold" if bold else "regular"
                        result_doc["pages"][p]["blocks"][b]["lines"][l]["words"][w]["color"]= color

        return result_doc
    
    def parse_doctr_to_hocr(self,doc):
        """Returns hocr with text attributes from doctr output"""

        hocr_content = []
        for p,page in enumerate(doc["pages"]):
            hocr_content.append('<div class="ocr_page">')
            for b,block in enumerate(page["blocks"]):
                block_geometry = block['geometry']
                hocr_content.append(f'<div class="ocr_carea" title="bbox {block_geometry[0][0]} {block_geometry[0][1]} {block_geometry[1][0]} {block_geometry[1][1]}">')
                for l,line in enumerate(block["lines"]):
                    line_geometry = line['geometry']
                    hocr_content.append(f'<div class="ocr_line" title="bbox {line_geometry[0][0]} {line_geometry[0][1]} {line_geometry[1][0]} {line_geometry[1][1]}">')
                    for w,word in enumerate(line["words"]):
                        word_geometry = word['geometry']
                        word_bbox = f'bbox {word_geometry[0][0]} {word_geometry[0][1]} {word_geometry[1][0]} {word_geometry[1][1]} x_wconf {word["confidence"]:.2f}'
                        word_text = f'{word["value"]}'

                        word_img = self.get_image(self.images[p], page["dimensions"], word_geometry)
                        bin_reg = self.get_image(self.bin_images[p], page["dimensions"], word_geometry)
                        processed_reg = self.get_image(self.processed_images[p], page["dimensions"], word_geometry)

                        bold = self.isbold(processed_reg, bin_reg,self.thres)
                        color = self.get_color(word_img, mode="dark")

                        b,g,r = color
                        style= f"font-weight: {'bold' if bold else 'regular'}; color: rgb({r},{g},{b});"
                        word_hocr = f'<span class="ocrx_word" title="{word_bbox}" style="{style}">{word_text}</span>'
                        hocr_content.append(word_hocr)
                    hocr_content.append('</div>')
                hocr_content.append('</div>')
            hocr_content.append('</div>')

        complete_hocr = f'<!DOCTYPE html><html><head><title></title></head><body>{" ".join(hocr_content)}</body></html>'

        return complete_hocr

    def parse_tesseract_to_hocr(self,hocr):
        """Returns hocr with text attributes from tesseract output"""

        soup = BeautifulSoup(hocr, 'html.parser')
        for p,page in enumerate(soup.find_all(class_="ocr_page")):
            for b,block in enumerate(page.find_all(class_="ocr_carea")):
                for par,para in enumerate(block.find_all(class_="ocr_par")):
                    for l,line in enumerate(para.find_all(class_="ocr_line")):
                        bbox, baseline, x_size, x_ascend, x_descend = line.get("title").split(";")
                        size, asc, dsc = float(x_size.split()[1]), float(x_ascend.split()[1]), float(x_descend.split()[1])
                        height = int(size + asc - dsc)
                        for w,word in enumerate(line.find_all(class_="ocrx_word")):
                            bbox,conf = word.get("title").split(";")
                            x1,y1,x2,y2 = map(int,bbox.split()[1:])

                            word_img = self.images[p][y1:y2,x1:x2]
                            bin_reg = self.bin_images[p][y1:y2,x1:x2]
                            processed_reg = self.processed_images[p][y1:y2,x1:x2]

                            bold = self.isbold(processed_reg,bin_reg,self.thres)
                            color = self.get_color(word_img,mode="dark")

                            b,g,r = color
                            fontsize = height/self.median_size
                            style = f"font-weight: {'bold' if bold else 'regular'}; color: rgb({r},{g},{b}); font-size: {fontsize}em"
                            word["style"] = style
        return soup.prettify()

    def parse_tesseract_to_json(self,hocr):
        """Returns hocr with text attributes from tesseract output"""

        soup = BeautifulSoup(hocr, 'html.parser')
        json_out = {"pages":[]}
        for p,page in enumerate(soup.find_all(class_="ocr_page")):
            props = page.get("title").split(";")
            page_bbox = props[-2]
            j_page = {"blocks":[],"bbox": page_bbox}
            for bl,block in enumerate(page.find_all(class_="ocr_carea")):
                j_block = {"block_id": block.get("id"),"bbox": block.get("title"),"paras":[]}
                for par,para in enumerate(block.find_all(class_="ocr_par")):
                    j_para = {"para_id":para.get("id"),"bbox": para.get("title"),"lines":[]}
                    for l,line in enumerate(para.find_all(class_="ocr_line")):
                        bbox, baseline, x_size, x_ascend, x_descend = line.get("title").split(";")
                        size, asc, dsc = float(x_size.split()[1]), float(x_ascend.split()[1]), float(x_descend.split()[1])
                        height = int(size + asc - dsc)
                        j_line = {"line_id":line.get("id"),"bbox": bbox,"line_height": height, "words":[]}
                        for w,word in enumerate(line.find_all(class_="ocrx_word")):
                            bbox,conf = word.get("title").split(";")

                            x1,y1,x2,y2 = map(int,bbox.split()[1:])
                            word_img = self.images[p][y1:y2,x1:x2]
                            bin_reg = self.bin_images[p][y1:y2,x1:x2]
                            processed_reg = self.processed_images[p][y1:y2,x1:x2]

                            bold = self.isbold(processed_reg,bin_reg,self.thres)
                            color = self.get_color(word_img,mode="dark")

                            b,g,r = color
                            fontsize = height/self.median_size
                            style = f"font-weight: {'bold' if bold else 'regular'}; color: rgb({r},{g},{b}); font-size: {fontsize}em"
                            word["style"] = style
                            j_word = {"word_id":word.get("id"),"bbox":bbox,"confidence":conf,"value":word.getText(),
                                      "font_weight":'bold' if bold else 'regular',"color":[r,g,b], "relative_font_size":fontsize}
                            j_line["words"].append(j_word)
                        j_para["lines"].append(j_line)
                    j_block["paras"].append(j_para)
                j_page["blocks"].append(j_block)
            json_out["pages"].append(j_page)
        return json_out

    def process_images_with_sizes(self,hocr):
        """Perform Morphological Opening with dynamic kernel sizes according to font sizes"""

        soup = BeautifulSoup(hocr, 'html.parser')
        sizes = []
        for p,page in enumerate(soup.find_all(class_="ocr_page")):
            subimages = {}
            for b,block in enumerate(page.find_all(class_="ocr_carea")):
                for par,para in enumerate(block.find_all(class_="ocr_par")):
                    for l,line in enumerate(para.find_all(class_="ocr_line")):

                        bbox, baseline, x_size, x_ascend, x_descend = line.get("title").split(";")
                        size, asc, dsc = float(x_size.split()[1]), float(x_ascend.split()[1]), float(x_descend.split()[1])
                        height = int(size + asc - dsc)
                        sizes.append(height)
                        word_count = len(line.find_all(class_="ocrx_word"))

                        if height not in subimages:
                            subimages[height] = {"image":np.ones_like(self.bin_images[p]) * 255, "words":0}
                        x1,y1,x2,y2 = map(int,bbox.split()[1:])
                        subimages[height]["image"][y1:y2,x1:x2] = self.bin_images[p][y1:y2,x1:x2]
                        subimages[height]["words"] += word_count

            max_words_size = None
            for size,im in subimages.items():
                if not max_words_size or im["words"] > subimages[max_words_size]["words"]:
                    max_words_size = size

            sw_k = self.stroke_width(subimages[max_words_size]["image"])
            sws = {}
            for size in subimages:
                sw = np.round(size / max_words_size * sw_k)
                sws[size] = sw

            res = np.ones_like(self.bin_images[p]) * 0
            for size in subimages:
                sub_processed = 255 - self.morph_open(subimages[size]["image"],int(sws[size])+1)
                res = cv2.bitwise_or(res, sub_processed)
            res = (255-res)
            self.processed_images[p] = res

        self.median_size = np.median(sizes)

    def binarize(self,image: np.ndarray)-> np.ndarray:
        """Returns binarized image"""

        return cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    def morph_open(self,binary_image: np.ndarray,k_size: int)-> np.ndarray:
        """Return image after morphological opening with specified kernel size """

        kernel = np.ones((k_size, k_size), np.uint8)
        invert = cv2.bitwise_not(binary_image)
        erosion = cv2.erode(invert, kernel,iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        processed = 255-dilation
        return processed
    
    def get_image(self, img, page_dim, doc_obj_geometry):
        """Returns subimage of given document object in the page"""

        page_dim = (page_dim[1], page_dim[0])
        box = np.array(doc_obj_geometry)
        box = (box*page_dim).astype(np.int64)
        region = img[box[0][1]:box[1][1],box[0][0]:box[1][0]]
        return region
    
    def isbold(self, processed, bin, thresh):
        """Return true if the image is bold else false"""

        bold_on = np.count_nonzero(processed==0)
        bin_on = np.count_nonzero(bin==0)
        if (bold_on+bin_on)==0:
            return False
        ratio = bold_on/(bold_on+bin_on)
        return ratio>=thresh
    
    def get_color(self, word, mode = "dark"):
        """Returns the forground color of the text in the given image of the word"""

        model = KMeans(n_clusters=2,n_init=1,init = [[0,0,0],[255,255,255]])
        model.fit(word.reshape(-1,3))
        colors = model.cluster_centers_.astype('int')
        if mode=="dark":
            color = self.get_darker_color(colors[0],colors[1])
        else:
            labels,counts = np.unique(model.labels_, return_counts=True)
            color_label = labels[np.argmin(counts)]
            color = colors[color_label].astype('int64')
        return color

    def get_darker_color(self,color1, color2):
        """Returns the darker color among given two colors"""

        luminance1 = 0.299 * color1[2] + 0.587 * color1[1] + 0.114 * color1[1]
        luminance2 = 0.299 * color2[2] + 0.587 * color2[1] + 0.114 * color2[0]
        if luminance1 < luminance2:
            return color1
        else:
            return color2
        
    def stroke_width(self,image):
        """Returns the stroke width of the text in the image"""
        pd = np.pad(255-image,2)
        pd[pd==255]=1
        distances = distance_transform_edt(pd)
        skeleton = skeletonize(pd)
        sk = np.zeros_like(pd)
        sk[skeleton]=255
        center_pixel_distances = distances[skeleton]
        stroke_width = np.mean(center_pixel_distances) * 2
        sw = np.round(stroke_width)
        return sw

def pdfToimg(in_file=None, out_file=None):
    """
    Convert PDF to images
    """
    if not in_file:
        return 0, "No PDF to Convert."
    if not out_file:
        return 0, "No destination"

    try:
        imgs = []
        pages = convert_from_path(in_file)
        for i, page in enumerate(pages):
            imgs.append(f"{out_file}_{i}.png")
            page.save(f"{out_file}_{i}.png", "PNG")
    except Exception as e:
        return None

    return imgs