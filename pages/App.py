import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import time

import cv2
from PIL import Image, ImageDraw

import easyocr
from paddleocr import PaddleOCR, draw_ocr
from mmocr.utils.ocr import MMOCR
import sys
from mycolorpy import colorlist as mcp
from PIL import ImageColor

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


@st.cache
def initializations():
    # the readers considered
    reader_type_list = ['EasyOCR', 'PPOCR']
    reader_type_dict = {'EasyOCR': 0, 'PPOCR': 1}

    # Columns for recognition details results
    cols_size = [2] + [2,1]*len(reader_type_list)

    # Colorscale for recognitions areas
    list_confid = list(np.round(np.arange(0,105.,0.1), 1))
    list_grad = mcp.gen_color_normalized(cmap="Greens",data_arr=np.array(list_confid))
    dict_back_colors = {list_confid[i]: list_grad[i] for i in range(len(list_confid))}

    # Color of the detected boxes
    color = (0, 76, 153)

    # Dicts of laguages supported by each reader
    dict_lang_easyocr = {'Abaza': 'abq', 'Adyghe': 'ady', 'Afrikaans': 'af', 'Angika': 'ang', \
    'Arabic': 'ar', 'Assamese': 'as', 'Avar': 'ava', 'Azerbaijani': 'az', 'Belarusian': 'be', \
    'Bulgarian': 'bg', 'Bihari': 'bh', 'Bhojpuri': 'bho', 'Bengali': 'bn', 'Bosnian': 'bs', \
    'Simplified Chinese': 'ch_sim', 'Traditional Chinese': 'ch_tra', 'Chechen': 'che', 'Czech': 'cs', \
    'Welsh': 'cy', 'Danish': 'da', 'Dargwa': 'dar', 'German': 'de', 'English': 'en', 'Spanish': 'es', \
    'Estonian': 'et', 'Persian (Farsi)': 'fa', 'French': 'fr', 'Irish': 'ga', 'Goan Konkani': 'gom', \
    'Hindi': 'hi', 'Croatian': 'hr', 'Hungarian': 'hu', 'Indonesian': 'id', 'Ingush': 'inh', \
    'Icelandic': 'is', 'Italian': 'it', 'Japanese': 'ja', 'Kabardian': 'kbd', 'Kannada': 'kn', \
    'Korean': 'ko', 'Kurdish': 'ku', 'Latin': 'la', 'Lak': 'lbe', 'Lezghian': 'lez', \
    'Lithuanian': 'lt', 'Latvian': 'lv', 'Magahi': 'mah', 'Maithili': 'mai', 'Maori': 'mi', \
    'Mongolian': 'mn', 'Marathi': 'mr', 'Malay': 'ms', 'Maltese': 'mt', 'Nepali': 'ne', \
    'Newari': 'new', 'Dutch': 'nl', 'Norwegian': 'no', 'Occitan': 'oc', 'Pali': 'pi', 'Polish': 'pl', \
    'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru', 'Serbian (cyrillic)': 'rs_cyrillic', \
    'Serbian (latin)': 'rs_latin', 'Nagpuri': 'sck', 'Slovak': 'sk', 'Slovenian': 'sl', \
    'Albanian': 'sq', 'Swedish': 'sv', 'Swahili': 'sw', 'Tamil': 'ta', 'Tabassaran': 'tab', \
    'Telugu': 'te', 'Thai': 'th', 'Tajik': 'tjk', 'Tagalog': 'tl', 'Turkish': 'tr', 'Uyghur': 'ug', \
    'Ukranian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi',
    }

    dict_lang_ppocr = {'Abaza': 'abq', 'Adyghe': 'ady', 'Afrikaans': 'af', 'Albanian': 'sq', 'Angika': 'ang', \
    'Arabic': 'ar', 'Avar': 'ava', 'Azerbaijani': 'az', 'Belarusian': 'be', 'Bhojpuri': 'bho', 'Bihari': 'bh', \
    'Bosnian': 'bs', 'Bulgarian': 'bg', 'Chinese & English': 'ch', 'Chinese Traditional': 'chinese_cht', \
    'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dargwa': 'dar', 'Dutch': 'nl', 'English': 'en', \
    'Estonian': 'et', 'French': 'fr', 'German': 'german', 'Goan Konkani': 'gom', 'Hindi': 'hi', \
    'Hungarian': 'hu', 'Icelandic': 'is', 'Indonesian': 'id', 'Ingush': 'inh', 'Irish': 'ga', \
    'Italian': 'it', 'Japan': 'japan', 'Kabardian': 'kbd', 'Korean': 'korean', 'Kurdish': 'ku', \
    'Lak': 'lbe', 'Latvian': 'lv', 'Lezghian': 'lez', 'Lithuanian': 'lt', 'Magahi': 'mah', \
    'Maithili': 'mai', 'Malay': 'ms', 'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', \
    'Mongolian': 'mn', 'Nagpur': 'sck', 'Nepali': 'ne', 'Newari': 'new', 'Norwegian': 'no', \
    'Occitan': 'oc', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', \
    'Russia': 'ru', 'Saudi Arabia': 'sa', 'Serbian(cyrillic)': 'rs_cyrillic', \
    'Serbian(latin)': 'rs_latin', 'Slovak': 'sk', 'Slovenian': 'sl', 'Spanish': 'es', \
    'Swahili': 'sw', 'Swedish': 'sv', 'Tabassaran': 'tab', 'Tagalog': 'tl', 'Tamil': 'ta', \
    'Telugu': 'te', 'Turkish': 'tr', 'Ukranian': 'uk', 'Urdu': 'ur', 'Uyghur': 'ug', \
    'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy',
    }

    list_dict_lang = [dict_lang_easyocr, dict_lang_ppocr]

    # Initialization of detection form
    st.session_state.columns_size = [1 for x in reader_type_list]
    st.session_state.column_width = [400 for x in reader_type_list]
    st.session_state.columns_color = ["rgb(0,0,0)" for x in reader_type_list]

    # Confidence color scale
    list_confid = list(np.arange(0,101,1))
    list_grad = mcp.gen_color_normalized(cmap="Greens",data_arr=np.array(list_confid))
    dict_back_colors = {list_confid[i]: list_grad[i] for i in range(len(list_confid))}

    list_y = [1 for i in list_confid]
    df = pd.DataFrame({'% confidence scale': list_confid, 'y': list_y})

    fig = px.scatter(df, x='% confidence scale', y='y', hover_data={'% confidence scale': True, 'y': False},
                                color=dict_back_colors.values(), range_y=[0.9,1.1], range_x=[0,100],
                                color_discrete_map="identity", height=50, symbol='y', symbol_sequence=['square'])
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, range=[0.1, 1.1], visible=False)
    fig.update_traces(marker_size=50)
    fig.update_layout(paper_bgcolor="white", margin=dict(b=0,r=0,t=0,l=0), xaxis_side="top", showlegend=False)

    return reader_type_list, reader_type_dict, color, list_dict_lang, cols_size, dict_back_colors, fig
###
@st.cache(show_spinner=False)
def init_readers(list_params):

    # Instantiations of the readers :
    # - EasyOCR
    reader_easyocr = easyocr.Reader([list_params[0][0]])

    # - PPOCR
    # Paddleocr 
    reader_ppocr = PaddleOCR(lang=list_params[1][0],
                             **list_params[1][1]
                           )

    list_readers = [reader_easyocr, reader_ppocr]

    return list_readers

###
@st.cache(show_spinner=False)
def load_image(image_file):
    image_path = "img."+image_file.name.split('.')[-1]
    img = Image.open(image_file)
    
    img_saved = img.save(image_path)

    # Read image
    image_orig = Image.open(image_path)
    image_cv2 = cv2.imread(image_path)
    return image_path, image_orig, image_cv2


###
def process_detect(image_path, list_images, list_readers, list_params):

    ## ------- EasyOCR Text detection

    easyocr_coordinates = text_detect('easyocr', list_readers[0], image_path, list_params[0][1])
    # The format of the coordinate is as follows: [x_min, x_max, y_min, y_max]

    # Format boxes coordinates for draw
    easyocr_boxes_coordinates = list(map(easyocr_coord_convert, easyocr_coordinates))

    # Visualization
    easyocr_image_detect = draw_detected(list_images[0], easyocr_boxes_coordinates, color, 'None', 7)

    ## ------- PPOCR Text detection

    ppocr_boxes_coordinates = text_detect('ppocr', list_readers[1], image_path, {})

    # Visualization
    ppocr_image_detect = draw_detected(list_images[0], ppocr_boxes_coordinates, color, 'None', 7)

    #
    list_images += [easyocr_image_detect, ppocr_image_detect]
    list_coordinates = [easyocr_boxes_coordinates, ppocr_boxes_coordinates]
    #

    return list_images, list_coordinates

###
def text_detect(reader_type, reader, image_path, dict_param):
# Return : boxes with text coordinates

    if reader_type == 'easyocr':
        # EasyOCR detection method
        # https://medium.com/quantrium-tech/integrating-multiple-ocr-models-to-perform-detection-and-recognition-separately-using-python-f2c73743e1e0 :
        # I have put some hyper parameter values that optimises the detection process based on my experiments.
        # The parameter width_ths specifies the maximum distance (horizontal) between two bounding boxes to be merged
        # (default threshold is 0.5) and mag_ratio magnifies the image based on the factor given.
        # Generally, you provide the factor >1 to enlarge and <1 to compress the image (default ratio is 1).

        detection_result = reader.detect(image_path,
                                         #width_ths=0.7, 
                                         #mag_ratio=1.5
                                         **dict_param
                                        )
        boxes_coordinates = detection_result[0][0]
        
    elif reader_type == 'ppocr':
        # PPOCR detection method        
        boxes_coordinates = reader.ocr(image_path, rec=False)
    
    return boxes_coordinates

###
def draw_detected(image, boxes_coordinates, color, posit='None', thickness=4):
# Input  : boxes coordinates, from top to bottom and from left to right 
#          [ [ [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max] ],
#            [ ...                                                            ]
#          ]  
# Return : image with detected zones
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, box in enumerate(boxes_coordinates):        
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, color, thickness)
        if posit != 'None':
            if posit == 'top_left':
                pos = tuple(box[0][0])
            elif posit == 'top_right':
                pos = tuple(box[1][0])
            image = cv2.putText(image,str(i+1), pos, font, 5.5,color,thickness,cv2.LINE_AA)
    
    image_drawn = Image.fromarray(image)

    return image_drawn

###
def easyocr_coord_convert(list_coord):
# Input :
#    list_coord (list) : [x_min, x_max, y_min, y_max]
# Output :
#    (list) : [ [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max] ]

    c = list_coord
    return [ [c[0], c[2]], [c[1], c[2]], [c[1], c[3]], [c[0], c[3]] ]

###
def text_recognize(list_readers, image_cv, boxes_coordinates, list_dict_params, dict_back_colors):
    df_results = pd.DataFrame([])
    list_images = []
    list_text_easyocr = []
    list_confidence_easyocr = []
    list_text_ppocr = []
    list_confidence_ppocr = []

    reader_easyocr = list_readers[0]
    reader_ppocr = PaddleOCR(**list_dict_params[1])

    for box in boxes_coordinates:
        box_ar = np.array(box).astype(np.int64)
        #x_min = np.array(box_ar)[:, 0].min()
        #x_max = np.array(box_ar)[:, 0].max()
        #y_min = np.array(box_ar)[:, 1].min()
        #y_max = np.array(box_ar)[:, 1].max()
        x_min = box_ar[:, 0].min()
        x_max = box_ar[:, 0].max()
        y_min = box_ar[:, 1].min()
        y_max = box_ar[:, 1].max()
        cropped = image_cv[y_min:y_max, x_min:x_max]
        list_images.append(cropped)
        
        # recognize with EasyOCR        
        #result = reader.readtext(cropped, **list_dict_params[0])
        result = reader_easyocr.recognize(cropped, **list_dict_params[0])
        try:
            list_text_easyocr.append(result[0][1])
            list_confidence_easyocr.append(np.round(100*result[0][2], 1))
        except:
            list_text_easyocr.append('Not recognize')
            list_confidence_easyocr.append(100.)

        # recognize with PPOCR        
        result = reader_ppocr.ocr(cropped, det=False, cls=False)
        try:
            list_text_ppocr.append(result[0][0])
            list_confidence_ppocr.append(np.round(100*result[0][1], 1))
        except:
            list_text_ppocr.append('Not recognize')
            list_confidence_ppocr.append(100.)


    df_results = pd.DataFrame({'cropped_image': list_images,
                               'text_easyocr': list_text_easyocr,
                               'confidence_easyocr': list_confidence_easyocr,
                               'text_ppocr': list_text_ppocr,
                               'confidence_ppocr': list_confidence_ppocr
                              }
                             )

    # Draw images with results
    list_reco_images = draw_reco_images(image_cv, boxes_coordinates, \
                                        [list_text_easyocr, list_text_ppocr], \
                                        [list_confidence_easyocr, list_confidence_ppocr], \
                                        dict_back_colors)

    return df_results, list_reco_images

###
def draw_reco_images(image, boxes_coordinates, list_texts, list_confid, dict_back_colors, \
                     font_scale=3, conf_threshold=64):
    img = image
    nb_readers = len(list_texts)
    list_reco_images = [img.copy() for i in range(nb_readers)]
    
    for num, box_ in enumerate(boxes_coordinates):
        box = np.array(box_).astype(np.int64)
        
        # For each box : draw the results of each recognizer
        for i in range(nb_readers):
            confid = np.round(list_confid[i][num], 0)
            rgb_color = ImageColor.getcolor(dict_back_colors[confid], "RGB")
            if confid < conf_threshold:
                text_color = (0, 0, 0)
            else:
                text_color = (255, 255, 255)

            list_reco_images[i] = cv2.rectangle(list_reco_images[i], (box[0][0], box[0][1]), \
                                                (box[2][0], box[2][1]), rgb_color, -1)
            list_reco_images[i] = cv2.putText(list_reco_images[i], list_texts[i][num], \
                                            (box[0][0],int(np.round((box[0][1]+box[2][1])/2,0))), \
                                            cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, 4)
    return list_reco_images

###
def update_font_scale(nb_col, dict_draw_reco):
    list_reco_images = draw_reco_images(**dict_draw_reco, \
                                        font_scale=st.session_state.font_scale_sld, \
                                        conf_threshold=st.session_state.conf_threshold_sld)

    with show_reco.container():
        reco_columns = st.columns(nb_col, gap='medium')
        column_width = 400
        for i, col in enumerate(reco_columns):                
            column_title = '<p style="font-size: 20px;color:rgb(0,0,0);">Recognition with ' + \
                            reader_type_list[i]+ '</p>'
            col.markdown(column_title, unsafe_allow_html=True)
            col.image(list_reco_images[i], width=column_width, use_column_width=True)  



####################################################################################################
##   MAIN 
####################################################################################################

##----------- Initializations -----------------------------------------------------------------------


st.title("OCR solutions comparator")
st.markdown("##### EasyOCR, PPOCR")

# Initializations
reader_type_list, reader_type_dict, color, list_dict_lang, cols_size, dict_back_colors, fig_colorscale = initializations()

##----------- Choose language & image -----------------------------------------------------------------
st.markdown("#### Choose languages for the text recognition")
lang_col = st.columns(2)
easyocr_key_lang = lang_col[0].selectbox(reader_type_list[0]+" :", list_dict_lang[0].keys(), 26)
easyocr_lang = list_dict_lang[0][easyocr_key_lang]
ppocr_key_lang = lang_col[1].selectbox(reader_type_list[1]+" :", list_dict_lang[1].keys(), 22)
ppocr_lang = list_dict_lang[1][ppocr_key_lang]

image_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

##----------- Process input image ----------------------------------------------------------------------
if image_file is not None:
    image_path, image_orig, image_cv2 = load_image(image_file)
    list_images = [image_orig, image_cv2]

##----------- Form with original image & hyperparameters for detectors ----------------------------------
    with st.form("form1"):
        with st.expander("Original image", expanded=True):
            col1, col2 = st.columns(2, gap="medium")
            col1.image(list_images[0], width=500, use_column_width=True)

            col2.markdown("#### Choose hyperparameters values for the detection:")
            tabs = col2.tabs(reader_type_list)
            with tabs[0]:
                t0_min_size = st.slider("min_size", 1, 20, 10, step=1, \
                              help="min_size (int, default = 10) - \
                              Filter text box smaller than minimum value in pixel")
                t0_text_threshold = st.slider("text_threshold", 0.1, 1., 0.7, step=0.1, \
                              help="text_threshold (float, default = 0.7) - Text confidence threshold")
                t0_low_text = st.slider("low_text", 0.1, 1., 0.4, step=0.1, \
                              help="low_text (float, default = 0.4) - Text low-bound score")
                t0_link_threshold = st.slider("link_threshold", 0.1, 1., 0.4, step=0.1, \
                              help="link_threshold (float, default = 0.4) - Link confidence threshold")
                t0_canvas_size = st.slider("canvas_size", 2000, 5000, 2560, step=10, \
                              help='''canvas_size (int, default = 2560) \n
Maximum image size. Image bigger than this value will be resized down''')
                t0_mag_ratio = st.slider("mag_ratio", 0.1, 5., 1., step=0.1, \
                              help="mag_ratio (float, default = 1) - Image magnification ratio")
                t0_slope_ths = st.slider("slope_ths", 0.01, 1., 0.1, step=0.01, \
                              help='''slope_ths (float, default = 0.1) - Maximum slope \
                              (delta y/delta x) to considered merging. \n
Low value means tiled boxes will not be merged.''')
                t0_ycenter_ths = st.slider("ycenter_ths", 0.1, 1., 0.5, step=0.1, \
                              help='''ycenter_ths (float, default = 0.5) - Maximum shift in y direction. \n
Boxes with different level should not be merged.''')
                t0_height_ths = st.slider("height_ths", 0.1, 1., 0.5, step=0.1, \
                              help='''height_ths (float, default = 0.5) - Maximum different in box height. \n
Boxes with very different text size should not be merged.''')
                t0_width_ths = st.slider("width_ths", 0.1, 1., 0.5, step=0.1, \
                              help="width_ths (float, default = 0.5) - Maximum horizontal \
                              distance to merge boxes.")
                t0_add_margin = st.slider("add_margin", 0.1, 1., 0.1, step=0.1, \
                              help='''add_margin (float, default = 0.1) - \
                              Extend bounding boxes in all direction by certain value. \n
This is important for language with complex script (E.g. Thai).''')
                t0_optimal_num_chars = st.slider("optimal_num_chars", None, 100, None, step=10, \
                              help="optimal_num_chars (int, default = None) - If specified, bounding \
                              boxes with estimated number of characters near this value are returned first.")

            with tabs[1]:
                t1_det_algorithm = st.selectbox('det_algorithm', ['DB'], \
                		help='Type of detection algorithm selected. (default = DB)')		
                t1_det_max_side_len = st.slider('det_max_side_len', 500, 2000, 960, step=10, \
                        help='''The maximum size of the long side of the image. (default = 960)\n
Limit the maximum image height and width.\n
When the long side exceeds this value, the long side will be resized to this size, and the short side \
will be scaled proportionally.''')
                t1_det_db_thresh =  st.slider('det_db_thresh', 0.1, 1., 0.3, step=0.1, \
                        help='''Binarization threshold value of DB output map. (default = 0.3) \n
Used to filter the binarized image of DB prediction, setting 0.-0.3 has no obvious effect on the result.''')
                t1_det_db_box_thresh = st.slider('det_db_box_thresh', 0.1, 1., 0.6, step=0.1, \
                        help='''The threshold value of the DB output box. (default = 0.6) \n
DB post-processing filter box threshold, if there is a missing box detected, it can be reduced as appropriate. \n
Boxes score lower than this value will be discard.''')
                t1_det_db_unclip_ratio = st.slider('det_db_unclip_ratio', 1., 3.0, 1.6, step=0.1, \
                        help='''The expanded ratio of DB output box. (default = 1.6) \n
Indicates the compactness of the text box, the smaller the value, the closer the text box to the text.''')
                t1_det_east_score_thresh = st.slider('det_east_cover_thresh', 0.1, 1., 0.8, step=0.1, \
                        help="Binarization threshold value of EAST output map. (default = 0.8)")
                t1_det_east_cover_thresh = st.slider('det_east_cover_thresh', 0.1, 1., 0.1, step=0.1, \
                        help='''The threshold value of the EAST output box. (default = 0.1) \n
Boxes score lower than this value will be discarded.''')
                t1_det_east_nms_thresh = st.slider('det_east_nms_thresh', 0.1, 1., 0.2, step=0.1, \
                        help="The NMS threshold value of EAST model output box. (default = 0.2)")
                t1_det_db_score_mode = st.selectbox('det_db_score_mode', ['fast', 'slow'], \
                        help='''slow: use polygon box to calculate bbox score, fast: use rectangle box \
                        to calculate. (default = fast) \n
Use rectangular box to calculate faster, and polygonal box more accurate for curved text area.''')


        submit_detect = st.form_submit_button("Launch detection")

##----------- Process text detection --------------------------------------------------------------------
    if submit_detect:
        # Process text detection

        if t0_optimal_num_chars == 0:
            t0_optimal_num_chars = None

        list_params_det = [[easyocr_lang, {'min_size': t0_min_size, 'text_threshold': t0_text_threshold, \
                       'low_text': t0_low_text, 'link_threshold': t0_link_threshold, \
                       'canvas_size': t0_canvas_size, 'mag_ratio': t0_mag_ratio, \
                       'slope_ths': t0_slope_ths, 'ycenter_ths': t0_ycenter_ths, \
                       'height_ths': t0_height_ths, 'width_ths': t0_width_ths, \
                       'add_margin': t0_add_margin, 'optimal_num_chars': t0_optimal_num_chars, }], \
                        [ppocr_lang, {'det_algorithm': t1_det_algorithm, 'det_max_side_len': t1_det_max_side_len, \
                        'det_db_thresh': t1_det_db_thresh, 'det_db_box_thresh': t1_det_db_box_thresh, \
                        'det_db_unclip_ratio': t1_det_db_unclip_ratio, 'det_east_score_thresh': t1_det_east_score_thresh, \
                        'det_east_cover_thresh': t1_det_east_cover_thresh, 'det_east_nms_thresh': t1_det_east_nms_thresh, \
                        'det_db_score_mode': t1_det_db_score_mode}]
                        ]
        with st.spinner("Readers initializations in progress (it may take a while) ..."):
            list_readers = init_readers(list_params_det)

        with st.spinner("Detection in progress ..."):
            list_images, list_coordinates = process_detect(image_path, list_images, \
                                                            list_readers, list_params_det)
            if 'list_readers' not in st.session_state:
                st.session_state.list_readers = list_readers
            if 'list_coordinates' not in st.session_state:
                st.session_state.list_coordinates = list_coordinates
            if 'list_images' not in st.session_state:
                st.session_state.list_images = list_images
            if 'list_params_det' not in st.session_state:
                st.session_state.list_params_det = list_params_det

    if 'list_coordinates' in st.session_state:
        list_coordinates = st.session_state.list_coordinates
        list_images = st.session_state.list_images
        list_readers = st.session_state.list_readers
        list_params_det = st.session_state.list_params_det

##----------- Text detection results --------------------------------------------------------------------
        st.subheader("Text detection")
        show_detect = st.empty()

        with show_detect.container():
            columns = st.columns(st.session_state.columns_size, gap='medium')
            for i, col in enumerate(columns):
                column_title = '<p style="font-size: 20px;color:'+st.session_state.columns_color[i] + \
                               ';">Detection with ' + reader_type_list[i]+ '</p>'
                col.markdown(column_title, unsafe_allow_html=True)
                col.image(list_images[i+2], width=st.session_state.column_width[i], use_column_width=True)

        st.subheader("Text recognition")

##----------- Form with detection results & hyperparameters for recognition -----------------------------
        with st.form("form2"):
            st.markdown("##### Using detection performed above by:")
            st.radio('Choose the detecter:', reader_type_list, key='detect_reader', \
                                             horizontal=True)
            with st.expander("Choose hyperparameters values for the recognition:", expanded=False):
                st.markdown("Choose hyperparameters values for the recognition")
                tabs = st.tabs(reader_type_list)
                with tabs[0]:
                    t0_decoder = st.selectbox('decoder', ['greedy', 'beamsearch', 'wordbeamsearch'], \
                        help="decoder (string, default = 'greedy') - options are 'greedy', 'beamsearch' \
                        and 'wordbeamsearch.")
                    t0_beamWidth = st.slider('beamWidth', 2, 20, 5, step=1, \
                        help="beamWidth (int, default = 5) - How many beam to keep when decoder = \
                        'beamsearch' or 'wordbeamsearch'.")
                    t0_batch_size = st.slider('batch_size', 1, 10, 1, step=1, \
                        help="batch_size (int, default = 1) - batch_size>1 will make EasyOCR faster \
                        but use more memory.")
                    t0_workers = st.slider('workers', 0, 10, 0, step=1, \
                        help="workers (int, default = 0) - Number thread used in of dataloader.")
                    t0_allowlist = st.text_input('allowlist', value="", max_chars=None, \
                        placeholder='Force EasyOCR to recognize only this subset of characters', \
                        help='''allowlist (string) - Force EasyOCR to recognize only subset of characters.\n
        Useful for specific problem (E.g. license plate, etc.)''')
                    t0_blocklist = st.text_input('blocklist', value="", max_chars=None, \
                        placeholder='Block subset of character (will be ignored if allowlist is given)', \
                        help='''blocklist (string) - Block subset of character. This argument will be \
                        ignored if allowlist is given.''')
                    t0_detail = st.radio('detail', [0, 1], 1, horizontal=True, \
                        help="detail (int, default = 1) - Set this to 0 for simple output")
                    t0_paragraph = st.radio('paragraph', [True, False], 1, horizontal=True, \
                        help='paragraph (bool, default = False) - Combine result into paragraph')
                    t0_contrast_ths = st.slider('contrast_ths', 0.05, 1., 0.1, step=0.01, \
                        help='''contrast_ths (float, default = 0.1) - Text box with contrast lower than \
                        this value will be passed into model 2 times.\n
        First is with original image and second with contrast adjusted to 'adjust_contrast' value.\n
        The one with more confident level will be returned as a result.''')
                    t0_adjust_contrast = st.slider('adjust_contrast', 0.1, 1., 0.5, step=0.1, \
                        help = 'adjust_contrast (float, default = 0.5) - target contrast level for low \
                        contrast text box')

                with tabs[1]:
                    t1_rec_algorithm = st.selectbox('rec_algorithm', ['CRNN', 'SVTR_LCNet'], 0, \
                        help="Type of recognition algorithm selected. (default=CRNN)")
                    t1_rec_batch_num = st.slider('rec_batch_num', 1, 50, step=1, \
                        help="When performing recognition, the batchsize of forward images. (default=30)")
                    t1_max_text_length = st.slider('max_text_length', 3, 250, 25, step=1, \
                        help="The maximum text length that the recognition algorithm can recognize. (default=25)")
                    t1_use_space_char = st.radio('use_space_char', [True, False], 0, horizontal=True, \
                        help="Whether to recognize spaces. (default=TRUE)")
                    t1_drop_score = st.slider('drop_score', 0., 1., 0.25, step=.05, \
                        help="Filter the output by score (from the recognition model), and those below \
                        this score will not be returned. (default=0.5)")

            submit_reco = st.form_submit_button("Launch recognition")

        if submit_reco:
##----------- Hightlight the detecter --------------------------------------
            with show_detect.container():
                columns_size = [1 for x in reader_type_list]
                column_width  = [400 for x in reader_type_list]
                columns_color = ["rgb(0,0,0)" for x in reader_type_list]
                columns_size[reader_type_dict[st.session_state.detect_reader]] = 2
                column_width[reader_type_dict[st.session_state.detect_reader]] = 500
                columns_color[reader_type_dict[st.session_state.detect_reader]] = "rgb(228,26,28)"
                columns = st.columns(columns_size, gap='medium')

                for i, col in enumerate(columns):
                    column_title = '<p style="font-size: 20px;color:'+columns_color[i] + \
                                    ';">Detection with ' + reader_type_list[i]+ '</p>'
                    col.markdown(column_title, unsafe_allow_html=True)
                    col.image(list_images[i+2], width=column_width[i])
                st.session_state.columns_size = columns_size
                st.session_state.column_width = column_width
                st.session_state.columns_color = columns_color

##----------- Process recognition ------------------------------------------
            reader_ind = reader_type_dict[st.session_state.detect_reader]
            list_boxes = list_coordinates[reader_ind]

            list_params_rec = [{'decoder': t0_decoder, 'beamWidth': t0_beamWidth, \
                                'batch_size': t0_batch_size, 'workers': t0_workers, \
                                'allowlist': t0_allowlist, 'blocklist': t0_blocklist, \
                                'detail': t0_detail, 'paragraph': t0_paragraph, \
                                'contrast_ths': t0_contrast_ths, 'adjust_contrast': t0_adjust_contrast
                               },
                               { **list_params_det[1][1], **{'rec_algorithm': t1_rec_algorithm, \
                               'rec_batch_num': t1_rec_batch_num, 'max_text_length': t1_max_text_length, \
                                'use_space_char': t1_use_space_char, 'drop_score': t1_drop_score}, \
                                **{'lang': list_params_det[1][0]}
                               } 
                              ]

            with st.spinner("Recognition in progress ..."):
                df_results, list_reco_images = text_recognize(list_readers, list_images[1], \
                                                              list_boxes, list_params_rec, dict_back_colors)
            st.session_state.df_results = df_results
            st.session_state.list_reco_images = list_reco_images
            st.session_state.list_boxes = list_boxes

        if 'df_results' in st.session_state:
##----------- Show recognition results ------------------------------------------------------------------
            results_cols = st.session_state.df_results.columns
            list_col_text = np.arange(1, len(cols_size), 2)
            list_col_confid = np.arange(2, len(cols_size), 2)

            dict_draw_reco = {'image': st.session_state.list_images[1], \
                              'boxes_coordinates': st.session_state.list_boxes, \
                              'list_texts': [st.session_state.df_results[x].to_list() \
                                                for x in results_cols[list_col_text]], \
                              'list_confid': [st.session_state.df_results[x].to_list() \
                                                 for x in results_cols[list_col_confid]], \
                              'dict_back_colors': dict_back_colors
                             }
            show_reco = st.empty()

            with st.form("form3"):
                st.plotly_chart(fig_colorscale, use_container_width=True)

                col_font, col_threshold = st.columns(2)
            
                col_font.slider('Font scale', 1, 7, 4, step=1, key="font_scale_sld")
                col_threshold.slider('% confidence threshold for text color change', 40, 100, 64, \
                                     step=1, key="conf_threshold_sld")
                col_threshold.write("(text color is black below this % confidence threshold, and white above)")
            
                with show_reco.container():
                    reco_columns = st.columns(len(reader_type_list), gap='medium')
                    column_width = 400
                    for i, col in enumerate(reco_columns):                
                        column_title = '<p style="font-size: 20px;color:rgb(0,0,0);">Recognition with ' + \
                                        reader_type_list[i]+ '</p>'
                        col.markdown(column_title, unsafe_allow_html=True)
                        col.image(st.session_state.list_reco_images[i], width=column_width, use_column_width=True)
                
                submit_resize = st.form_submit_button("Refresh")

            if submit_resize:
                update_font_scale(len(reader_type_list), dict_draw_reco)

            st.subheader("Recognition details")
            with st.expander("Detailed areas", expanded=True):
                cols = st.columns(cols_size)
                cols[0].markdown('#### Detected area')
                for i in range(1, len(reader_type_list)*2, 2):
                    cols[i].markdown('#### with ' + reader_type_list[i//2])

                for row in st.session_state.df_results.itertuples():
                    #cols = st.columns(1 + len(reader_type_list)*2)
                    cols = st.columns(cols_size)
                    cols[0].image(row.cropped_image, width=150)
                    for i in range(1, len(cols), 2):
                        cols[i].write(getattr(row, results_cols[i]))
                        cols[i+1].write("("+str(getattr(row, results_cols[i+1]))+"%)")
