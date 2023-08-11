import os
import subprocess


files = os.listdir('./../../docs/')


for file in files:
    input = './../../docs/' + file
    output = file.split('.')[0]
    subprocess.call(['python', 'main.py', '-i', input, '-o', output, '-l', 'Devanagari', '--ocr_only'])