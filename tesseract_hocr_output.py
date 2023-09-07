# INSTALLING DEPENDENCIES
# !sudo apt-get install tesseract-ocr-[hin]
# ! apt install tesseract-ocr
# !pip install pytesseract
# ! apt install libtesseract-dev
# !pip install pdf2image
# !apt-get install poppler-utils


# IMPORTING LIBRARIES
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
import cv2
from pdf2image import convert_from_path
import numpy as np


# GETTING TESSERACT OBJECTS

def get_tesseract_objs(img_path):
  img=cv2.imread(img_path)
  tesseract_output=pytesseract.image_to_data(img,lang='hin+eng',config='--psm 6').split('\n')
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
        line_dict[block][para][line]=[x,y,w,h,text]
      else:
        if para not in line_dict[block].keys():
          line_dict[block][para]={}
          line_dict[block][para][line]=[x,y,w,h,text]
        else:
          if line not in line_dict[block][para].keys():
            line_dict[block][para][line]=[x,y,w,h,text]
          else:
            line_dict[block][para][line][0]=min(line_dict[block][para][line][0],x)
            line_dict[block][para][line][1]=min(line_dict[block][para][line][1],y)
            line_dict[block][para][line][2]=x+w
            line_dict[block][para][line][3]=max(line_dict[block][para][line][3],h)
            line_dict[block][para][line][4]+=' '+text
  for block in line_dict.keys():
    for para in line_dict[block].keys():
      for line in line_dict[block][para].keys():
        word=line_dict[block][para][line]
        objs.append([word[0],word[1],word[2]+word[0],word[3]+word[1],'Text',word[4]])
        # print([word[0],word[1],word[2]+word[0],word[3]+word[1],'Text',word[4]])
  objs=sorted(objs,key=lambda x:x[1])
  return objs


# HOCR OUTPUT
def hocr_output(file):

  # getting the pages from the pdfs
  images = convert_from_path(file)
  body=''
  page_depth=0
  for page_index in range(len(images)):
    images[page_index].save('page'+file.split('/')[-1].split('.')[0]+ str(page_index) +'.jpg', 'JPEG')

    #  getting the tesseract objects
    result=get_tesseract_objs('page'+file.split('/')[-1].split('.')[0]+ str(page_index) +'.jpg')
    img=cv2.imread('page'+file.split('/')[-1].split('.')[0]+ str(page_index) +'.jpg')
    temp_depth=0

    # making the hocr output
    # bbox = [ xmin , ymin , xmax , ymax , type , content]
    for index,bbox in enumerate(result):
      # print(bbox)
      span=f"\t\t\t<span style='position:absolute;width:{str(bbox[2]-bbox[0])}px;top: {str(page_depth+bbox[1])}px;left: {str(bbox[0])}px;'>"
      # checking if the object is text or image or table
      if bbox[4]=='Text':
        line=f"<p style='font_size=0.5em;'>{bbox[5]}</p>"
        span+=line

      elif bbox[4]=='Image' or bbox[4]=='Table':
        crp=img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        cv2.imwrite('crp_'+file.split('/')[-1].split('.')[0]+str(index)+'.jpg',crp)
        i=f'<img src="'+'crp_'+file.split('/')[-1].split('.')[0]+str(index)+'.jpg"> '
        span+=i

      span+='</span>\n'
      body+=span
      temp_depth=bbox[3]
    page_depth+=temp_depth+200

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

  # saving the hocr output
  with open(f"{file.split('/')[-1].split('.')[0]}_hocr_output.html",'w') as file:
    file.write(html)


#  to run the program insert the file path
# run this hocr_output(<your file path>)