#INSTALLING DEPENDECIES

# !git clone https://github.com/mindee/doctr.git
# !pip install -e doctr/.[tf]

# !pip install pdf2image
# !apt-get install poppler-utils
# !pip install opencv-python
# !pip install doctr

# IMPORTING DEPENDENCIES


import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np
from pdf2image import convert_from_path

# HOCR OUTPUT

def hocr_output(file):

  #getting the doctr output for the pdf
  model=ocr_predictor(pretrained=True)
  document=DocumentFile.from_pdf(file)
  doctr_output=model(document)
  json_response=doctr_output.export()

  #getting the images of pages from the pdf
  images =   convert_from_path(file)
  for page_idx in range(len(images)):
    images[page_idx].save('page'+file.split('/')[-1].split('.')[0]+ str(page_idx) +'.jpg', 'JPEG')
  hocr=''
  page_depth=0

  #iterating over the output of doctr and preprocessing the doctr results for mapping the line output and stroing all the results in the result list
  for page in json_response['pages']:
    y,x=page['dimensions']
    page_index=page['page_idx']
    doctr_objs=[]
    for block in page['blocks']:
      ((xmin,ymin),(xmax,ymax))=block['geometry']
      xmin,ymin,xmax,ymax=[int(xmin*x),int(ymin*y),int(xmax*x),int(ymax*y)]
      typ='Text'
      content=[]
      prev=0
      temp_content=''
      for line in block['lines']:
        ytemp=int(line['geometry'][0][1]*y)
        if abs(ytemp-prev)>7:
          if len(temp_content)>2:
            content.append(temp_content)
            temp_content=''
        prev=ytemp
        for word in line['words']:
          temp_content+=word['value']+' '
      content.append(temp_content)
      # output saved in the format of xmin, ymin, xmax, ymax, type, content
      doctr_objs.append([xmin,ymin,xmax,ymax,typ,content])

    #sorting the result on the basis of their ymin order
    def basis(x):
      return x[1]
    doctr_objs=sorted(doctr_objs,key=basis)
    for i in doctr_objs:
      print(i)

    # making the html file using result
    img=cv2.imread('page'+file.split('/')[-1].split('.')[0]+ str(page_index) +'.jpg')
    for index,bbox in enumerate(doctr_objs):
      print(bbox)
      span=f"\t\t\t<span style='position:absolute;width:{str(bbox[2]-bbox[0])}px;top: {str(page_depth+bbox[1])}px;left: {str(bbox[0])}px;'>"
      #if type == text or image or table
      if bbox[4]=='Text':
        for i in bbox[5]:
          p=f"<p style='font_size=0.5em;'>{i}</p>"
          span+=p
      elif bbox[4]=='Image' or bbox[4]=='Table':

        crp=img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        cv2.imwrite(f'crp{index}'+file.split('/')[-1].split('.')[0]+'.jpg',crp)
        i=f'<img src="crp{index}'+file.split('/')[-1].split('.')[0]+'.jpg"> '
        span+=i
      span+='</span>\n'
      hocr+=span
    
    # setting the depth of present page to start from the next page
    page_depth+=y

  # creating the hocr output and save it
  html=f'''
  <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta http-equiv="X-UA-Compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>{file.split('/')[-1].split('.')[0]}</title>
    </head>
    <body>
  {hocr}
    </body>
  </html>'''
  with open(f"{file.split('/')[-1].split('.')[0]}_hocr_output.html",'w') as f:
    f.write(html)

# run this hocr_output(<your file path>)