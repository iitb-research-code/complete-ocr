# pip install lxml
# pip install beautifulsoup4
# How to run => python3 xml2hocr.py input_filename.xml output_filename

from bs4 import BeautifulSoup, Doctype
import sys

# Read the XML file
with open(sys.argv[1]) as xml_file:
    soup = BeautifulSoup(xml_file, 'xml')

# Declaration
xml_declaration = '<?xml version="1.0" encoding="utf-8"?>'
doctype_declaration = ' html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"'
soup.insert(0, Doctype(doctype_declaration))

fetch_html = soup.find('html')
lang_attribute = fetch_html['xml:lang']
xmlns_attribute = fetch_html['xmlns']
new_html = soup.new_tag('html', lang=f'{lang_attribute}', xmlns=f'{xmlns_attribute}')

# Head-Tag
head_tag = soup.find('head')
title_tag = head_tag.find('title')
head = soup.new_tag('head')

# Meta-Tags
# Find the existing <meta> tags
fetch_doctr_ver = soup.find('meta', attrs={'name': 'ocr-system'})
doctr_ver = fetch_doctr_ver['content']

fetch_ocr_capabilities = soup.find('meta', attrs={'name': 'ocr-capabilities'})
ocr_capabilities = fetch_ocr_capabilities['content']

# Create new <meta> tags with the desired content
meta_content_type = soup.new_tag('meta', content="text/html; charset=utf-8")
meta_ocr_system = soup.new_tag('meta', content=f"{doctr_ver}")
meta_ocr_capabilities = soup.new_tag('meta', content=f"{ocr_capabilities}")

# Set the 'name' attribute separately
meta_ocr_system['name'] = "ocr-system"
meta_ocr_capabilities['name'] = "ocr-capabilities"

# Remove the existing <meta> tags from the head section
for existing_meta in soup.find_all('meta'):
    existing_meta.extract()

# Append the new tags to the head section
head_tag.append(title_tag)
head_tag.append(meta_content_type)
head_tag.append(meta_ocr_system)
head_tag.append(meta_ocr_capabilities)

# Code to remove unwanted spaces from "bbox" attribute values
# Find all elements with "bbox" attribute
elements_with_bbox = soup.find_all(attrs={"title": lambda x: x and 'bbox' in x})

# Remove unwanted spaces
for element in elements_with_bbox:
    bbox_value = element['title']
    bbox_value = " ".join(bbox_value.split())
    element['title'] = bbox_value.strip()

# Write the changes to the new file
output_file = sys.argv[2]
with open(f'{output_file}.hocr', 'w', encoding='utf-8') as f:
    f.write(soup.prettify())
