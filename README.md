# sampoorn-ocr
Repository hosting all the developed OCR models for various Indic languages

The Code is written to contain Printed and Handwritten models

# Installations
'''
pip install -r requirements.txt
pip install "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"

'''

# Run The OCR Engine
'''
python main.py -i <INPUT_FILE>/
-it <INPUT_TYPE> -o <PROJECT_FOLDER_NAME> -l <LANGUAGE_MODEL> -t <OCR_METHOD: True IF HANDWRITTEN , False IF PRINTED > -c <OCR_ONLY : True or False>
'''
For Further Help run
'''
python main.py -h
'''