import re
from bs4 import BeautifulSoup, Tag


def postprocess(soup):
    for soup_div_child in soup.find_all("div"):
        try:
                if len(soup_div_child.find_next("span")) <= 3: #and len(re.findall("\w|\d", soup_div_child.get_text()))!=0:
                    prev_div = soup_div_child.find_previous("div")
                    prev_p = soup_div_child.find_previous("p", attrs="ocr_par")
                    prev_span = soup_div_child.find_previous("span", attrs="ocr_line")
                    curr_span = soup_div_child.find_next("span", attrs="ocr_line")
                    curr_span_bbox = curr_span.attrs["title"].split(";")[0]
                    prev_div_bbox = prev_div.attrs["title"].split(";")[0]
                    prev_p_bbox = prev_p.attrs["title"].split(";")[0]
                    
                    cx1, cy1, cx2, cy2 = map(int, curr_span_bbox.split(" ")[1:])
                    pdx1, pdy1, pdx2, pdy2 = map(int, prev_div_bbox.split(" ")[1:])
                    ppx1, ppy1, ppx2, ppy2 = map(int, prev_p_bbox.split(" ")[1:])
                    ndx1, ndy1, ndx2, ndy2 = min(cx1, pdx1), min(cy1, pdy1), max(cx2, pdx2), max(cy2, pdy2)     # for div class
                    npx1, npy1, npx2, npy2 = min(cx1, ppx1), min(cy1, ppy1), max(cx2, ppx2), max(cy2, ppy2)     # for p class
                    
                    # change the coordinates of the bbox of the div
                    new_bbox_parent_div = " ".join(["bbox"]+[str(ndx1), str(ndy1), str(ndx2), str(ndy2)])
                    new_bbox_parent_p = " ".join(["bbox"]+[str(npx1), str(npy1), str(npx2), str(npy2)])
            
                    prev_div.attrs["title"] = new_bbox_parent_div
                    prev_p.attrs["title"] = new_bbox_parent_p
                    
                    # inserting the target span tags after the last span tag
                    # soup_div_child.find_previous("span").insert_after(soup_div_child.find_next("span"))
                    prev_span.insert_after(curr_span)
                    
                    
        except Exception as e:
                # print(e)
                continue

    # removing the leftover div class       
    for soup_div in soup.div:
        for el in soup_div.next_element:
            if isinstance(el, Tag):
                if len(list(el.children)) < 3:
                    el.parent.decompose()
                    
    return soup
        
        
def hocr_correction(hocr):
    soup = BeautifulSoup(hocr, "html.parser")
    return postprocess(soup)