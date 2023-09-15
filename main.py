import os
import string
import pandas as pd
import argparse
from PIL import Image
import pytesseract
from tqdm import tqdm

from pdf2image import convert_from_path

from src.config import DPI, JPEGOPT, TESSDATA_DIR_CONFIG, OUTPUT_DIR



### For simple filename generation
def simple_counter_generator(prefix="", suffix=""):
    i=0
    while True:
        i+=1
        yield 'p'
        

def extract_info_mode1(txt):
    lines = txt.split('\n')
    eng, ind, des = '', '', ''
    data = []
    for line in lines:
        if(len(line)>4 and line[0] not in string.punctuation and not line[0].isdigit()):
            if(ord(line[0])<150 and len(line.strip().split(' ')[0])>=2):
                if(len(ind)>3):
                    data.append([eng, ind, des])
                eng, ind, des = '', '', ''
                line = line.strip().split()
                eng_word_flag=True
                for word in line:
                    if(word in '-|:;"\'`~!@#$%^&*()-_+?<>[{()}]=1234567890'):
                        continue
                    elif(ord(word[0])<120 and eng_word_flag) :
                        eng += word + ' '
                    else:
                        ind += word + ' '
                        eng_word_flag=False
            else:
                des += line
    
    if(len(ind)>3):
        data.append([eng, ind, des])
        
    return data

def extract_info_mode2(txt):
    lines=txt.split('\n')
    data=[]
    sanskrit_word,english_vocab,meaning='','',''
    prev_sans_word=0
    for word in lines:
        word=word.split()
        try:
            if word[11]!='-1' and word[11].strip()[0] not in '-|:;"\'`~!@#$%^&*()-_+?<>[{()}]=1234567890':
                # print(word[11])
                if int(word[6])<500:
                    if abs(int(word[7])-prev_sans_word)<10:
                        sanskrit_word+=word[11]+' '
                    else:
                        if len(sanskrit_word)>0 and len(english_vocab)>0:
                            data.append([sanskrit_word,english_vocab,meaning])
                        sanskrit_word,english_vocab,meaning=word[11],'',''
                    prev_sans_word=int(word[7])
                elif int(word[6])<900:
                    english_vocab+=word[11]+' '
                else:
                    meaning+=word[11]+' '
        except:
            pass
    if len(sanskrit_word)>0 and len(english_vocab)>0:
        data.append([sanskrit_word,english_vocab,meaning])

    return data



def main(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    
    images_folder = os.path.join(OUTPUT_DIR, args.images_folder_name, 'images')
    
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        output_file=simple_counter_generator("page",".jpg")
        convert_from_path(args.orig_pdf_path ,output_folder=images_folder, dpi=DPI,fmt='jpeg',jpegopt=JPEGOPT,output_file=output_file)
 
 
    # images_folder = './outputs/test/images/'
    img_files = sorted(os.listdir(images_folder))
    
    
    result = []
    
    curr_page = 1
    for img_file in tqdm(img_files):
        
        if(args.start_page is not None and curr_page < int(args.start_page)):
            curr_page += 1
            continue
        
        elif(args.end_page is not None and curr_page > int(args.end_page)):
            break
        
        else:
            curr_page += 1
            
        img_path = os.path.join(images_folder, img_file)
        image = Image.open(img_path)
        gray_image = image.convert('L')
        
        if args.mode=='1':
            txt = pytesseract.image_to_string(gray_image, lang=args.language_model, config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode1(txt)
                result.extend(value)
            except:
                pass

        elif args.mode=='2':
            txt = pytesseract.image_to_data(gray_image,lang=args.language_model,config=TESSDATA_DIR_CONFIG)

            try:
                value = extract_info_mode2(txt)
                result.extend(value)
            except:
                pass
        
    if args.mode == '1':
        df = pd.DataFrame(result, columns = ['English Word','Indic Meaning','Indic Description'])
    elif args.mode == '2':
        df = pd.DataFrame(result, columns = ['Sanskrit Word','English Vocab','Description'])
    
    out_file = os.path.join(OUTPUT_DIR, args.images_folder_name, 'results.csv')
    df.to_csv(out_file, index=False)
    
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Documents OCR Input Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--orig_pdf_path", type=str, default=None, help="path to the input pdf file")
    parser.add_argument("-im", "--images_folder_name", type=str, default="pdf", help="type of input file | pdf/images")
    parser.add_argument("-l", "--language_model", type=str, default="Devangari", help="language to be used for OCR")
    parser.add_argument("-m", "--mode", type=str, default='1', help="mode 1,2,3 => 1 and for Eng-Sanskrit => 2")
    parser.add_argument("-s", "--start-page", type=str, default=None, help="Start page for OCR")
    parser.add_argument("-e", "--end-page", type=str, default=None, help="End page for OCR")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

