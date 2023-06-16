import numpy as np
import cv2
import torch
import glob as glob
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytesseract
from ocr_config import faster_rcnn_model_path, det_threshold, table_recognition_language, row_determining_threshold, col_determining_threshold


# Returns Faster RCNN model to perfrom table cell-wise detection
def create_model(num_classes):
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

# Returns true if two rectangles b1 and b2 overlap, b is of the form [x1, y1, x2, y2]
def do_overlap(b1, b2):
    if (b1[0] >= b2[2]) or (b1[2] <= b2[0]) or (b1[3] <= b2[1]) or (b1[1] >= b2[3]):
         return False
    else:
        return True

# Get cells which are a part of a table
def get_cells_from_table(tab, cells):
    tablecells = []
    for c in cells:
        overlap = do_overlap(tab, c)
        if overlap:
            tablecells.append(c)
    return tablecells

def get_tables_from_page(img_file):
    full_table_response = []
    # set the computation device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # load the model and the trained weights
    model = create_model(num_classes=3).to(device)
    model.load_state_dict(torch.load(
        faster_rcnn_model_path, map_location=device
    ))
    model.eval()

    # classes: 0 index is reserved for background
    CLASSES = [
        'bkg', 'table', 'cell'
    ]
    # any detection having score below this will be discarded
    detection_threshold = det_threshold

    # get the image file name for saving output later on
    image_name = img_file
    image = cv2.imread(image_name)
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        print('Table Prediction Complete')

        # Trim classes for top k boxes predicted over threshold score
        classes = pred_classes[:len(boxes)]

        # Collect table and cells 
        tables = []
        cells = []
        for i in range(len(boxes)):
            if classes[i] == 'table':
                tables.append(boxes[i])
            else:
                cells.append(boxes[i])

        for tablebbox in tables:
        
            tabcells = get_cells_from_table(tablebbox, cells)
            # print(tablebbox)
            # print(len(tabcells))

            # Proceed only if table has cells
            if len(tabcells) > 0:
                # Sort cells based on y coordinates
                strcells = sorted(tabcells, key=lambda b:b[1]+b[3], reverse=False)

                # Calculate Mean height
                cell_heights = [c[3] - c[1] for c in tabcells]
                mean_height = int(np.mean(cell_heights))

                # Assign row to each cell based on y coordinate wise arrangement 
                cellrow = [0]
                assign_row = 0
                for i in range(len(strcells) - 1):
                    consec_cell_height = strcells[i + 1][1] - strcells[i][1]
                    if consec_cell_height > row_determining_threshold * mean_height:
                        assign_row = assign_row + 1
                    cellrow.append(assign_row)

                # Get number of rows and columns
                rows = list(set(cellrow))
                nrows = len(rows)
                counts = [0] * nrows
                for cr in cellrow:
                    counts[cr] = counts[cr] + 1
                ncols = max(counts)

                # Generate row-wise cell sequences of bounding boxes
                cellrows = {}
                for i in rows:
                    row_wise_cells = []
                    for j in range(len(strcells)):
                        if i == cellrow[j]:
                            row_wise_cells.append(strcells[j])
                    row_wise_cells = sorted(row_wise_cells, key=lambda b:b[0], reverse=False)
                    cellrows[i] = row_wise_cells

                tableresponse = {}
                tableresponse['bbox'] = tablebbox 
                tableresponse['nrows'] = nrows
                tableresponse['ncells'] = len(strcells)
                tableresponse['cellrows'] = cellrows
                full_table_response.append(tableresponse)
    return full_table_response

def get_cell_text(image, cellbbox, lang):
    cropped_image = image[cellbbox[1]:cellbbox[3], cellbbox[0]:cellbbox[2]]
    text = pytesseract.image_to_string(cropped_image, lang=lang, config='--psm 6')
    return text

def get_merged_cell(final_cells):
    if len(final_cells) == 1:
        return final_cells[0]
    x1 = [c[0] for c in final_cells]
    y1 = [c[1] for c in final_cells]
    x2 = [c[2] for c in final_cells]
    y2 = [c[3] for c in final_cells]
    cell = [min(x1), min(y1), max(x2), max(y2)]
    return cell

def get_final_cell(tablecellrows, skeleton, rowindex, colindex):
    final_cells = []
    row_to_consider = tablecellrows[rowindex]
    col_skeleton = skeleton[rowindex]
    assert len(row_to_consider) == len(col_skeleton)
    for i in range(len(col_skeleton)):
        if col_skeleton[i] == colindex:
            final_cells.append(row_to_consider[i])
    if len(final_cells) == 0:
        return []
    else:
        final_cell = get_merged_cell(final_cells)
    return final_cell

def get_hocr_from_table_response(imgfile, tableresponse):
    raw_image = cv2.imread(imgfile)
    nrows = tableresponse['nrows']
    tablecellrows = tableresponse['cellrows']
    tablebbox = tableresponse['bbox']
    lang = table_recognition_language

    # Preparing skeleton to assign column numbers
    final_skeleton = []
    max_entries_per_row = []
    for row in tablecellrows:
        row_to_consider = tablecellrows[row]
        # Calculate Mean height
        cell_widths = [c[3] - c[1] for c in row_to_consider]
        mean_width = int(np.mean(cell_widths))
        # Sort cells in same row from left to right
        ltor_cells = sorted(row_to_consider, key=lambda b:b[0], reverse=False)
        # Assign col number to every cell
        col_to_assign = 0
        assigned_col = [0]
        for i in range(len(ltor_cells) - 1):
            consec_cell_diff = ltor_cells[i + 1][0] - ltor_cells[i][0]
            if consec_cell_diff > col_determining_threshold * mean_width:
                col_to_assign = col_to_assign + 1
            assigned_col.append(col_to_assign)
        max_entries_per_row.append(col_to_assign)
        final_skeleton.append(assigned_col)
        
    ncols = max(max_entries_per_row)  + 1
    # print(final_skeleton)
    # print("Rows => " + str(nrows))
    # print("Cols => " + str(ncols))

    # HOCR Generation
    hocr = '<table class="ocr_tab" border=1 style="margin: 0px auto; text-align: center;"'
    tabbbox = " ".join(tablebbox.astype(str))
    hocr = hocr + f' title = "bbox {tabbbox}" >'
    for i in range(nrows):
        hocr = hocr + '<tr>'
        for j in range(ncols):
            cell = get_final_cell(tablecellrows, final_skeleton, i, j)
            if len(cell) == 0:
                text = ''
            else :
                text = get_cell_text(raw_image, cell, lang)
            hocr = hocr + '<td>' + text + '</td>' 
        hocr = hocr + '</tr>'
    hocr = hocr + '</table>'
    entry = []
    entry.append(hocr)
    entry.append(tablebbox)
    return entry

def get_final_table_hocrs_from_image(imgfile):
    full_table_response = get_tables_from_page(imgfile)
    full_hocrs_response = []
    for table_response in full_table_response:
        hocr = get_hocr_from_table_response(imgfile, table_response)
        full_hocrs_response.append(hocr)
    return full_hocrs_response

img_file = '../DEMO/10.jpg'
print(get_final_table_hocrs_from_image(img_file))
