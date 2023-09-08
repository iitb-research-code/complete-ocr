
OUTPUT_DIR = './../../output_books/'

jpg_options = {
    "quality"    : 100,
    "progressive": True,
    "optimize"   : False
}

directories = ['Images', 'Comments', 'VerifierOutput', 'CorrectorOutput', 'Inds', 'Dicts', 'CroppedImages', 'MaskedImages']


model_config = 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config'
extra_config = "MODEL.ROI_HEADS.SCORE_THRESH_TEST"
label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

storeMaskedImages = True

tessdata_dir_config = r'--tessdata-dir "./../../../tessdata/"'


# Table OCR Related Properties
model_path = './../../models/table_detection_model.pth'
det_threshold = 0.5
table_recognition_language = 'eng'
row_determining_threshold = 0.6667
col_determining_threshold = 0.5
nms_table_threshold = 0.1
nms_cell_threshold = 0.0001
