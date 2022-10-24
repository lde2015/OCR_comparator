"""_summary_

    Returns:
        _type_: _description_
"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

import cv2
from PIL import Image, ImageColor
import PIL
import easyocr
from paddleocr import PaddleOCR
from mmocr.utils.ocr import MMOCR
import pytesseract
from pytesseract import Output
import os
from mycolorpy import colorlist as mcp

@st.cache
def convert_df(in_df):
    """_summary_

    Args:
        in_df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return in_df.to_csv().encode('utf-8')

###
def easyocr_coord_convert(in_list_coord):
    """_summary_

    Args:
        in_list_coord (_type_): _description_

    Returns:
        _type_: _description_
    """
# Input :
#    list_coord (list) : [x_min, x_max, y_min, y_max]
# Output :
#    (list) : [ [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max] ]

    coord = in_list_coord
    return [[coord[0], coord[2]], [coord[1], coord[2]], [coord[1], coord[3]], [coord[0], coord[3]]]

@st.cache(show_spinner=False)
def initializations():
    """_summary_

    Returns:
        _type_: _description_
    """
    # the readers considered
    out_reader_type_list = ['EasyOCR', 'PPOCR', 'MMOCR', 'Tesseract']
    out_reader_type_dict = {'EasyOCR': 0, 'PPOCR': 1, 'MMOCR': 2, 'Tesseract': 3}

    # Columns for recognition details results
    out_cols_size = [2] + [2,1]*(len(out_reader_type_list)-1) # Except Tesseract

    # Color of the detected boxes
    out_color = (0, 76, 153)

    # Dicts of laguages supported by each reader
    out_dict_lang_easyocr = {'Abaza': 'abq', 'Adyghe': 'ady', 'Afrikaans': 'af', 'Angika': 'ang', \
    'Arabic': 'ar', 'Assamese': 'as', 'Avar': 'ava', 'Azerbaijani': 'az', 'Belarusian': 'be', \
    'Bulgarian': 'bg', 'Bihari': 'bh', 'Bhojpuri': 'bho', 'Bengali': 'bn', 'Bosnian': 'bs', \
    'Simplified Chinese': 'ch_sim', 'Traditional Chinese': 'ch_tra', 'Chechen': 'che', \
    'Czech': 'cs', 'Welsh': 'cy', 'Danish': 'da', 'Dargwa': 'dar', 'German': 'de', \
    'English': 'en', 'Spanish': 'es', 'Estonian': 'et', 'Persian (Farsi)': 'fa', 'French': 'fr', \
    'Irish': 'ga', 'Goan Konkani': 'gom', 'Hindi': 'hi', 'Croatian': 'hr', 'Hungarian': 'hu', \
    'Indonesian': 'id', 'Ingush': 'inh', 'Icelandic': 'is', 'Italian': 'it', 'Japanese': 'ja', \
    'Kabardian': 'kbd', 'Kannada': 'kn', 'Korean': 'ko', 'Kurdish': 'ku', 'Latin': 'la', \
    'Lak': 'lbe', 'Lezghian': 'lez', 'Lithuanian': 'lt', 'Latvian': 'lv', 'Magahi': 'mah', \
    'Maithili': 'mai', 'Maori': 'mi', 'Mongolian': 'mn', 'Marathi': 'mr', 'Malay': 'ms', \
    'Maltese': 'mt', 'Nepali': 'ne', 'Newari': 'new', 'Dutch': 'nl', 'Norwegian': 'no', \
    'Occitan': 'oc', 'Pali': 'pi', 'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', \
    'Russian': 'ru', 'Serbian (cyrillic)': 'rs_cyrillic', 'Serbian (latin)': 'rs_latin', \
    'Nagpuri': 'sck', 'Slovak': 'sk', 'Slovenian': 'sl', 'Albanian': 'sq', 'Swedish': 'sv', \
    'Swahili': 'sw', 'Tamil': 'ta', 'Tabassaran': 'tab', 'Telugu': 'te', 'Thai': 'th', \
    'Tajik': 'tjk', 'Tagalog': 'tl', 'Turkish': 'tr', 'Uyghur': 'ug', 'Ukranian': 'uk', \
    'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi'}

    out_dict_lang_ppocr = {'Abaza': 'abq', 'Adyghe': 'ady', 'Afrikaans': 'af', 'Albanian': 'sq', \
    'Angika': 'ang', 'Arabic': 'ar', 'Avar': 'ava', 'Azerbaijani': 'az', 'Belarusian': 'be', \
    'Bhojpuri': 'bho','Bihari': 'bh','Bosnian': 'bs','Bulgarian': 'bg','Chinese & English': 'ch', \
    'Chinese Traditional': 'chinese_cht', 'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', \
    'Dargwa': 'dar', 'Dutch': 'nl', 'English': 'en', 'Estonian': 'et', 'French': 'fr', \
    'German': 'german','Goan Konkani': 'gom','Hindi': 'hi','Hungarian': 'hu','Icelandic': 'is', \
    'Indonesian': 'id', 'Ingush': 'inh', 'Irish': 'ga', 'Italian': 'it', 'Japan': 'japan', \
    'Kabardian': 'kbd', 'Korean': 'korean', 'Kurdish': 'ku', 'Lak': 'lbe', 'Latvian': 'lv', \
    'Lezghian': 'lez', 'Lithuanian': 'lt', 'Magahi': 'mah', 'Maithili': 'mai', 'Malay': 'ms', \
    'Maltese': 'mt', 'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Nagpur': 'sck', \
    'Nepali': 'ne', 'Newari': 'new', 'Norwegian': 'no', 'Occitan': 'oc', 'Persian': 'fa', \
    'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russia': 'ru', 'Saudi Arabia': 'sa', \
    'Serbian(cyrillic)': 'rs_cyrillic', 'Serbian(latin)': 'rs_latin', 'Slovak': 'sk', \
    'Slovenian': 'sl', 'Spanish': 'es', 'Swahili': 'sw', 'Swedish': 'sv', 'Tabassaran': 'tab', \
    'Tagalog': 'tl', 'Tamil': 'ta', 'Telugu': 'te', 'Turkish': 'tr', 'Ukranian': 'uk', \
    'Urdu': 'ur', 'Uyghur': 'ug', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy'}

    out_dict_lang_mmocr = {'English & Chinese': 'en'}

    out_dict_lang_tesseract = {'Afrikaans': 'afr','Albanian': 'sqi','Amharic': 'amh', \
    'Arabic': 'ara', 'Armenian': 'hye','Assamese': 'asm','Azerbaijani - Cyrilic': 'aze_cyrl', \
    'Azerbaijani': 'aze', 'Basque': 'eus','Belarusian': 'bel','Bengali': 'ben','Bosnian': 'bos', \
    'Breton': 'bre', 'Bulgarian': 'bul','Burmese': 'mya','Catalan; Valencian': 'cat', \
    'Cebuano': 'ceb', 'Central Khmer': 'khm','Cherokee': 'chr','Chinese - Simplified': 'chi_sim', \
    'Chinese - Traditional': 'chi_tra','Corsican': 'cos','Croatian': 'hrv','Czech': 'ces', \
    'Danish':'dan','Dutch; Flemish':'nld','Dzongkha':'dzo','English, Middle (1100-1500)':'enm', \
    'English': 'eng','Esperanto': 'epo','Estonian': 'est','Faroese': 'fao', \
    'Filipino (old - Tagalog)': 'fil','Finnish': 'fin','French, Middle (ca.1400-1600)': 'frm', \
    'French': 'fra','Galician': 'glg','Georgian - Old': 'kat_old','Georgian': 'kat', \
    'German - Fraktur': 'frk','German': 'deu','Greek, Modern (1453-)': 'ell','Gujarati': 'guj', \
    'Haitian; Haitian Creole': 'hat','Hebrew': 'heb','Hindi': 'hin','Hungarian': 'hun', \
    'Icelandic': 'isl','Indonesian': 'ind','Inuktitut': 'iku','Irish': 'gle', \
    'Italian - Old': 'ita_old','Italian': 'ita','Japanese': 'jpn','Javanese': 'jav', \
    'Kannada': 'kan','Kazakh': 'kaz','Kirghiz; Kyrgyz': 'kir','Korean (vertical)': 'kor_vert', \
    'Korean': 'kor','Kurdish (Arabic Script)': 'kur_ara','Lao': 'lao','Latin': 'lat', \
    'Latvian':'lav','Lithuanian':'lit','Luxembourgish':'ltz','Macedonian':'mkd','Malay':'msa', \
    'Malayalam': 'mal','Maltese': 'mlt','Maori': 'mri','Marathi': 'mar','Mongolian': 'mon', \
    'Nepali': 'nep','Norwegian': 'nor','Occitan (post 1500)': 'oci', \
    'Orientation and script detection module':'osd','Oriya':'ori','Panjabi; Punjabi':'pan', \
    'Persian':'fas','Polish':'pol','Portuguese':'por','Pushto; Pashto':'pus','Quechua':'que', \
    'Romanian; Moldavian; Moldovan': 'ron','Russian': 'rus','Sanskrit': 'san', \
    'Scottish Gaelic': 'gla','Serbian - Latin': 'srp_latn','Serbian': 'srp','Sindhi': 'snd', \
    'Sinhala; Sinhalese': 'sin','Slovak': 'slk','Slovenian': 'slv', \
    'Spanish; Castilian - Old': 'spa_old','Spanish; Castilian': 'spa','Sundanese': 'sun', \
    'Swahili': 'swa','Swedish': 'swe','Syriac': 'syr','Tajik': 'tgk','Tamil': 'tam', \
    'Tatar':'tat','Telugu':'tel','Thai':'tha','Tibetan':'bod','Tigrinya':'tir','Tonga':'ton', \
    'Turkish': 'tur','Uighur; Uyghur': 'uig','Ukrainian': 'ukr','Urdu': 'urd', \
    'Uzbek - Cyrilic': 'uzb_cyrl','Uzbek': 'uzb','Vietnamese': 'vie','Welsh': 'cym', \
    'Western Frisian': 'fry','Yiddish': 'yid','Yoruba': 'yor'}

    out_list_dict_lang = [out_dict_lang_easyocr, out_dict_lang_ppocr, out_dict_lang_mmocr, \
                          out_dict_lang_tesseract]

    # Initialization of detection form
    if 'columns_size' not in st.session_state:
        st.session_state.columns_size = [1 for x in out_reader_type_list]
    if 'column_width' not in st.session_state:
        st.session_state.column_width = [400 for x in out_reader_type_list]
    if 'columns_color' not in st.session_state:
        st.session_state.columns_color = ["rgb(0,0,0)" for x in out_reader_type_list]

    # Confidence color scale
    out_list_confid = list(np.arange(0,101,1))
    out_list_grad = mcp.gen_color_normalized(cmap="Greens",data_arr=np.array(out_list_confid))
    out_dict_back_colors = {out_list_confid[i]: out_list_grad[i] \
                                                for i in range(len(out_list_confid))}

    list_y = [1 for i in out_list_confid]
    df_confid = pd.DataFrame({'% confidence scale': out_list_confid, 'y': list_y})

    out_fig = px.scatter(df_confid, x='% confidence scale', y='y', \
                hover_data={'% confidence scale': True, 'y': False},
                color=out_dict_back_colors.values(), range_y=[0.9,1.1], range_x=[0,100],
                color_discrete_map="identity",height=50,symbol='y',symbol_sequence=['square'])
    out_fig.update_xaxes(showticklabels=False)
    out_fig.update_yaxes(showticklabels=False, range=[0.1, 1.1], visible=False)
    out_fig.update_traces(marker_size=50)
    out_fig.update_layout(paper_bgcolor="white", margin=dict(b=0,r=0,t=0,l=0), xaxis_side="top", \
                          showlegend=False)

    return out_reader_type_list, out_reader_type_dict, out_color, out_list_dict_lang, \
           out_cols_size, out_dict_back_colors, out_fig

###
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def init_easyocr(in_params):
    """_summary_

    Args:
        in_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    out_ocr = easyocr.Reader(in_params)
    return out_ocr

###
@st.cache(show_spinner=False)
#@st.experimental_memo(show_spinner=False)
def init_ppocr(in_params):
    """_summary_

    Args:
        in_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    out_ocr = PaddleOCR(lang=in_params[0],
                    **in_params[1]
                   )
    return out_ocr

###
#@st.cache(show_spinner=False, hash_funcs={torch.nn.parameter.Parameter: lambda _: None}, \
#          allow_output_mutation=True)
@st.experimental_memo(show_spinner=False)
def init_mmocr(in_params):
    """_summary_

    Args:
        in_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    out_ocr = MMOCR(recog=None, **in_params[1])
    return out_ocr

###
def init_readers(in_list_params):
    """_summary_

    Args:
        in_list_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Instantiations of the readers :
    # - EasyOCR
    with st.spinner("EasyOCR reader initialization in progress ..."):
        reader_easyocr = init_easyocr([in_list_params[0][0]])

    # - PPOCR
    # Paddleocr
    with st.spinner("PPOCR reader initialization in progress ..."):
        reader_ppocr = init_ppocr(in_list_params[1])

    # - MMOCR
    with st.spinner("MMOCR reader initialization in progress ..."):
        reader_mmocr = init_mmocr(in_list_params[2])

    out_list_readers = [reader_easyocr, reader_ppocr, reader_mmocr]

    return out_list_readers

###
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def load_image(in_image_file):
    """_summary_

    Args:
        in_image_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    out_image_path = "img."+in_image_file.name.split('.')[-1]
    img = Image.open(in_image_file)
    img_saved = img.save(out_image_path)
    img = Image.open(in_image_file)

    img_saved = img.save(out_image_path)

    # Read image
    out_image_orig = Image.open(out_image_path)
    out_image_cv2 = cv2.imread(out_image_path)
    return out_image_path, out_image_orig, out_image_cv2

###
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def easyocr_detect(_in_reader, in_image_path, in_params):
    """_summary_

    Args:
        _in_reader (_type_): _description_
        in_image_path (_type_): _description_
        in_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    # EasyOCR detection method
    # https://medium.com/quantrium-tech/integrating-multiple-ocr-models-to-perform-detection-and-recognition-separately-using-python-f2c73743e1e0 :
    # I have put some hyper parameter values that optimises the detection process based on my experiments.
    # The parameter width_ths specifies the maximum distance (horizontal) between two bounding boxes to be merged
    # (default threshold is 0.5) and mag_ratio magnifies the image based on the factor given.
    # Generally, you provide the factor >1 to enlarge and <1 to compress the image (default ratio is 1).
    try:
        dict_param = in_params[1]
        detection_result = _in_reader.detect(in_image_path,
                                             #width_ths=0.7,
                                             #mag_ratio=1.5
                                             **dict_param
                                             )
        easyocr_coordinates = detection_result[0][0]

        # The format of the coordinate is as follows: [x_min, x_max, y_min, y_max]
        # Format boxes coordinates for draw
        out_easyocr_boxes_coordinates = list(map(easyocr_coord_convert, easyocr_coordinates))
        out_status = 'OK'
    except Exception as e:
        out_easyocr_boxes_coordinates = []
        out_status = e

    return out_easyocr_boxes_coordinates, out_status

###
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def ppocr_detect(_in_reader, in_image_path):
    """_summary_

    Args:
        _in_reader (_type_): _description_
        in_image_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # PPOCR detection method
    try:
        out_ppocr_boxes_coordinates = _in_reader.ocr(in_image_path, rec=False)
        out_status = 'OK'
    except Exception as e:
        out_ppocr_boxes_coordinates = []
        out_status = e

    return out_ppocr_boxes_coordinates, out_status

###
#@st.cache(show_spinner=False, hash_funcs={torch.nn.parameter.Parameter: lambda _: None})
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def mmocr_detect(_in_reader, in_image_path):
    """_summary_

    Args:
        _in_reader (_type_): _description_
        in_image_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # MMOCR detection method
    out_mmocr_boxes_coordinates = []
    try:
        det_result = _in_reader.readtext(in_image_path, details=True)
        bboxes_list = [res['boundary_result'] for res in det_result]
        for bboxes in bboxes_list:
            for bbox in bboxes:
                if len(bbox) > 9:
                    min_x = min(bbox[0:-1:2])
                    min_y = min(bbox[1:-1:2])
                    max_x = max(bbox[0:-1:2])
                    max_y = max(bbox[1:-1:2])
                    #box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
                else:
                    min_x = min(bbox[0:-1:2])
                    min_y = min(bbox[1::2])
                    max_x = max(bbox[0:-1:2])
                    max_y = max(bbox[1::2])
                box4 = [ [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y] ]
                out_mmocr_boxes_coordinates.append(box4)
        out_status = 'OK'
    except Exception as e:
        out_status = e

    return out_mmocr_boxes_coordinates, out_status

###
def cropped_1box(in_box, in_img):
    """_summary_

    Args:
        in_box (_type_): _description_
        in_img (_type_): _description_

    Returns:
        _type_: _description_
    """
    box_ar = np.array(in_box).astype(np.int64)
    x_min = box_ar[:, 0].min()
    x_max = box_ar[:, 0].max()
    y_min = box_ar[:, 1].min()
    y_max = box_ar[:, 1].max()
    out_cropped = in_img[y_min:y_max, x_min:x_max]

    return out_cropped

###
@st.experimental_memo(show_spinner=False)
def tesserocr_detect(_in_img, in_params):
    """_summary_

    Args:
        _in_img (_type_): _description_
        in_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        dict_param = in_params[1]
        df_res = pytesseract.image_to_data(_in_img, **dict_param, output_type=Output.DATAFRAME)
        df_res['box'] = df_res.apply(lambda d: [[d['left'], d['top']], \
                                                [d['left'] + d['width'], d['top']], \
                                                [d['left'] + d['width'], d['top'] + d['height']], \
                                                [d['left'], d['top'] + d['height']], \
                                               ], axis=1)
        out_tesserocr_boxes_coordinates = df_res[df_res.word_num > 0]['box'].to_list()
        out_status = 'OK'
    except Exception as e:
        out_tesserocr_boxes_coordinates = []
        out_status = e

    return out_tesserocr_boxes_coordinates, out_status

###
@st.experimental_memo(show_spinner=False)
def process_detect(in_image_path, _in_list_images, _in_list_readers, in_list_params, in_color):
    """_summary_

    Args:
        in_image_path (_type_): _description_
        _in_list_images (_type_): _description_
        _in_list_readers (_type_): _description_
        in_list_params (_type_): _description_
        in_color (_type_): _description_

    Returns:
        _type_: _description_
    """
    ## ------- EasyOCR Text detection
    with st.spinner('EasyOCR Text detection in progress ...'):
        easyocr_boxes_coordinates,easyocr_status = easyocr_detect(_in_list_readers[0], \
                                                                  in_image_path, in_list_params[0])
        # Visualization
        if easyocr_boxes_coordinates:
            easyocr_image_detect = draw_detected(_in_list_images[0], easyocr_boxes_coordinates, \
                                                 in_color, 'None', 7)
        else:
            easyocr_boxes_coordinates = easyocr_status
    ##

    ## ------- PPOCR Text detection
    with st.spinner('PPOCR Text detection in progress ...'):
        ppocr_boxes_coordinates, ppocr_status = ppocr_detect(_in_list_readers[1], in_image_path)
        # Visualization
        if ppocr_boxes_coordinates:
            ppocr_image_detect = draw_detected(_in_list_images[0], ppocr_boxes_coordinates, \
                                               in_color, 'None', 7)
        else:
            ppocr_image_detect = ppocr_status
    ##

    ## ------- MMOCR Text detection
    with st.spinner('MMOCR Text detection in progress ...'):
        mmocr_boxes_coordinates, mmocr_status = mmocr_detect(_in_list_readers[2], in_image_path)
        # Visualization
        if mmocr_boxes_coordinates:
            mmocr_image_detect = draw_detected(_in_list_images[0], mmocr_boxes_coordinates, \
                                               in_color, 'None', 7)
        else:
            mmocr_image_detect = mmocr_status
    ##

    ## ------- Tesseract Text detection
    with st.spinner('Tesseract Text detection in progress ...'):
        tesserocr_boxes_coordinates, tesserocr_status = tesserocr_detect(_in_list_images[0], \
                                                                         in_list_params[3])
        # Visualization
        if tesserocr_boxes_coordinates:
            tesserocr_image_detect = draw_detected(_in_list_images[0],tesserocr_boxes_coordinates, \
                                                   in_color, 'None', 7)
        else:
            tesserocr_image_detect = tesserocr_status
    ##
    #
    out_list_images = _in_list_images + [easyocr_image_detect, ppocr_image_detect, \
                                         mmocr_image_detect, tesserocr_image_detect]
    out_list_coordinates = [easyocr_boxes_coordinates, ppocr_boxes_coordinates, \
                            mmocr_boxes_coordinates, tesserocr_boxes_coordinates]
    #

    return out_list_images, out_list_coordinates

###
def draw_detected(in_image, in_boxes_coordinates, in_color, posit='None', in_thickness=4):
    """_summary_

    Args:
        in_image (_type_): _description_
        in_boxes_coordinates (_type_): _description_
        in_color (_type_): _description_
        posit (str, optional): _description_. Defaults to 'None'.
        in_thickness (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
# Input  : boxes coordinates, from top to bottom and from left to right
#          [ [ [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max] ],
#            [ ...                                                            ]
#          ]
# Return : image with detected zones
    work_img = in_image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for ind_box, box in enumerate(in_boxes_coordinates):
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        work_img = cv2.polylines(np.array(work_img), [box], True, in_color, in_thickness)
        if posit != 'None':
            if posit == 'top_left':
                pos = tuple(box[0][0])
            elif posit == 'top_right':
                pos = tuple(box[1][0])
            work_img = cv2.putText(work_img, str(ind_box+1), pos, font, 5.5, color, \
                                   in_thickness,cv2.LINE_AA)

    out_image_drawn = Image.fromarray(work_img)

    return out_image_drawn

##
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def get_cropped(in_boxes_coordinates, in_image_cv):
    """_summary_

    Args:
        in_boxes_coordinates (_type_): _description_
        in_image_cv (_type_): _description_

    Returns:
        _type_: _description_
    """
    out_list_images = []
    for box in in_boxes_coordinates:
        cropped = cropped_1box(box, in_image_cv)
        out_list_images.append(cropped)
    return out_list_images

###
def process_recog(in_list_readers, in_image_cv, in_boxes_coordinates, in_list_dict_params, \
                  in_dict_back_colors):
    """_summary_

    Args:
        in_list_readers (_type_): _description_
        in_image_cv (_type_): _description_
        in_boxes_coordinates (_type_): _description_
        in_list_dict_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    out_df_results = pd.DataFrame([])

    list_text_easyocr = []
    list_confidence_easyocr = []
    list_text_ppocr = []
    list_confidence_ppocr = []
    list_text_mmocr = []
    list_confidence_mmocr = []

    # Create cropped images from detection
    list_cropped_images = get_cropped(in_boxes_coordinates, in_image_cv)

    # Recognize with EasyOCR
    with st.spinner('EasyOCR Text recognition in progress ...'):
        list_text_easyocr, list_confidence_easyocr, status_easyocr = \
            easyocr_recog(list_cropped_images, in_list_readers[0], in_list_dict_params[0])
    ##

    # Recognize with PPOCR
    with st.spinner('PPOCR Text recognition in progress ...'):
        list_text_ppocr, list_confidence_ppocr, status_ppocr = \
            ppocr_recog(list_cropped_images, in_list_dict_params[1])
    ##

    # Recognize with MMOCR
    with st.spinner('MMOCR Text recognition in progress ...'):
        list_text_mmocr, list_confidence_mmocr, status_mmocr = \
            mmocr_recog(list_cropped_images, in_list_dict_params[2])
    ##

    # Recognize with Tesseract
    with st.spinner('Tesseract Text recognition in progress ...'):
        out_df_results_tesseract, status_tesseract = \
        tesserocr_recog(in_image_cv, in_list_dict_params[3], len(list_cropped_images))
    ##

    # Create results data frame
    out_df_results = pd.DataFrame({'cropped_image': list_cropped_images,
                                   'text_easyocr': list_text_easyocr,
                                   'confidence_easyocr': list_confidence_easyocr,
                                   'text_ppocr': list_text_ppocr,
                                   'confidence_ppocr': list_confidence_ppocr,
                                   'text_mmocr': list_text_mmocr,
                                   'confidence_mmocr': list_confidence_mmocr
                                  }
                                 )

    # Draw images with results
    out_list_reco_images = draw_reco_images(in_image_cv, in_boxes_coordinates, \
                           [list_text_easyocr, list_text_ppocr, list_text_mmocr], \
                           [list_confidence_easyocr,list_confidence_ppocr,list_confidence_mmocr], \
                           in_dict_back_colors, out_df_results_tesseract)

    out_list_reco_status = [status_easyocr, status_ppocr, status_mmocr, status_tesseract]

    return out_df_results, out_list_reco_images, out_df_results_tesseract, out_list_reco_status

##
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def easyocr_recog(in_list_images, _in_reader_easyocr, in_params):
    """_summary_

    Args:
        in_list_images (_type_): _description_
        _in_reader_easyocr (_type_): _description_
        in_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    progress_bar = st.progress(0)
    out_list_text_easyocr = []
    out_list_confidence_easyocr = []
    ## ------- EasyOCR Text recognition
    try:
        step = 0*len(in_list_images) # first recognition process
        nb_steps = 4 * len(in_list_images)
        for ind_img, cropped in enumerate(in_list_images):
            result = _in_reader_easyocr.recognize(cropped, **in_params)
            try:
                out_list_text_easyocr.append(result[0][1])
                out_list_confidence_easyocr.append(np.round(100*result[0][2], 1))
            except:
                out_list_text_easyocr.append('Not recognize')
                out_list_confidence_easyocr.append(100.)
            progress_bar.progress((step+ind_img+1)/nb_steps)
        out_status = 'OK'
    except Exception as e:
        out_status = e
    progress_bar.empty()

    return out_list_text_easyocr, out_list_confidence_easyocr, out_status

##
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def ppocr_recog(in_list_images, in_params):
    """_summary_

    Args:
        in_list_images (_type_): _description_
        in_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    ## ------- PPOCR Text recognition
    out_list_text_ppocr = []
    out_list_confidence_ppocr = []
    try:
        reader_ppocr = PaddleOCR(**in_params)
        step = 1*len(in_list_images) # second recognition process
        nb_steps = 4 * len(in_list_images)
        progress_bar = st.progress(step/nb_steps)

        for ind_img, cropped in enumerate(in_list_images):
            result = reader_ppocr.ocr(cropped, det=False, cls=False)
            try:
                out_list_text_ppocr.append(result[0][0])
                out_list_confidence_ppocr.append(np.round(100*result[0][1], 1))
            except:
                out_list_text_ppocr.append('Not recognize')
                out_list_confidence_ppocr.append(100.)
            progress_bar.progress((step+ind_img+1)/nb_steps)
        out_status = 'OK'
    except Exception as e:
        out_status = e
    progress_bar.empty()

    return out_list_text_ppocr, out_list_confidence_ppocr, out_status

##
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def mmocr_recog(in_list_images, in_params):
    """_summary_

    Args:
        in_list_images (_type_): _description_
        in_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    ## ------- MMOCR Text recognition
    out_list_text_mmocr = []
    out_list_confidence_mmocr = []
    try:
        reader_mmocr = MMOCR(det=None, **in_params)
        step = 2*len(in_list_images) # third recognition process
        nb_steps = 4 * len(in_list_images)
        progress_bar = st.progress(step/nb_steps)

        for ind_img, cropped in enumerate(in_list_images):
            result = reader_mmocr.readtext(cropped, details=True)
            try:
                out_list_text_mmocr.append(result[0]['text'])
                out_list_confidence_mmocr.append(np.round(100* \
                                                         (np.array(result[0]['score']).mean()), 1))
            except:
                out_list_text_mmocr.append('Not recognize')
                out_list_confidence_mmocr.append(100.)
            progress_bar.progress((step+ind_img+1)/nb_steps)
        out_status = 'OK'
    except Exception as e:
        out_status = e
    progress_bar.empty()

    return out_list_text_mmocr, out_list_confidence_mmocr, out_status

##
#@st.cache(show_spinner=False)
@st.experimental_memo(show_spinner=False)
def tesserocr_recog(in_img, in_params, in_nb_images):
    """_summary_

    Args:
        in_img (_type_): _description_
        in_params (_type_): _description_
        in_nb_images (_type_): _description_

    Returns:
        _type_: _description_
    """
    ## ------- Tesseract Text recognition
    step = 3*in_nb_images # second recognition process
    nb_steps = 4 * in_nb_images
    progress_bar = st.progress(step/nb_steps)

    try:
        out_df_result = pytesseract.image_to_data(in_img, **in_params,output_type=Output.DATAFRAME)

        out_df_result['box'] = out_df_result.apply(lambda d: [[d['left'], d['top']], \
                                                    [d['left'] + d['width'], d['top']], \
                                                    [d['left']+d['width'], d['top']+d['height']], \
                                                    [d['left'], d['top'] + d['height']], \
                                                    ], axis=1)
        out_df_result['cropped'] = out_df_result['box'].apply(lambda b: cropped_1box(b, in_img))
        out_df_result = out_df_result[(out_df_result.word_num > 0) & (out_df_result.text != ' ')] \
                             .reset_index(drop=True)
        out_status = 'OK'
    except Exception as e:
        out_df_result = pd.DataFrame([])
        out_status = e

    progress_bar.progress(1.)

    return out_df_result, out_status

###
def draw_reco_images(in_image, in_boxes_coordinates, in_list_texts, in_list_confid, \
                     in_dict_back_colors, in_df_results_tesseract, in_font_scale=3, \
                     in_conf_threshold=65):
    """_summary_

    Args:
        in_image (_type_): _description_
        in_boxes_coordinates (_type_): _description_
        in_list_texts (_type_): _description_
        in_list_confid (_type_): _description_
        in_df_results_tesseract (_type_): _description_
        in_font_scale (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """
    img = in_image.copy()
    nb_readers = len(in_list_texts) + 1 # Add Tesseract
    out_list_reco_images = [img.copy() for i in range(nb_readers)]

    for num, box_ in enumerate(in_boxes_coordinates):
        box = np.array(box_).astype(np.int64)

        # For each box : draw the results of each recognizer
        for ind_r in range(nb_readers-1):
            confid = np.round(in_list_confid[ind_r][num], 0)
            rgb_color = ImageColor.getcolor(in_dict_back_colors[confid], "RGB")
            if confid < in_conf_threshold:
                text_color = (0, 0, 0)
            else:
                text_color = (255, 255, 255)

            out_list_reco_images[ind_r] = cv2.rectangle(out_list_reco_images[ind_r], \
                                                       (box[0][0], box[0][1]), \
                                                       (box[2][0], box[2][1]), rgb_color, -1)
            out_list_reco_images[ind_r] = cv2.putText(out_list_reco_images[ind_r], \
                                                      in_list_texts[ind_r][num], \
                                            (box[0][0],int(np.round((box[0][1]+box[2][1])/2,0))), \
                                            cv2.FONT_HERSHEY_DUPLEX, in_font_scale, text_color, 2)

    # Add Tesseract process
    if not in_df_results_tesseract.empty:
        ind = nb_readers-1
        for num, box_ in enumerate(in_df_results_tesseract['box'].to_list()):
            box = np.array(box_).astype(np.int64)
            confid = np.round(in_df_results_tesseract.iloc[num]['conf'], 0)
            rgb_color = ImageColor.getcolor(in_dict_back_colors[confid], "RGB")
            if confid < in_conf_threshold:
                text_color = (0, 0, 0)
            else:
                text_color = (255, 255, 255)

            out_list_reco_images[ind] = cv2.rectangle(out_list_reco_images[ind], \
                                              (box[0][0], box[0][1]), \
                                              (box[2][0], box[2][1]), rgb_color, -1)
            out_list_reco_images[ind] = cv2.putText(out_list_reco_images[ind], \
                                            in_df_results_tesseract.iloc[num]['text'], \
                                            (box[0][0],int(np.round((box[0][1]+box[2][1])/2,0))), \
                                            cv2.FONT_HERSHEY_DUPLEX, in_font_scale, text_color, 4)

    return out_list_reco_images

###
def update_font_scale(in_nb_col, in_dict_draw_reco, in_reader_type_list):
    """_summary_

    Args:
        in_nb_col (_type_): _description_
        in_dict_draw_reco (_type_): _description_
        in_reader_type_list (_type_): _description_
    """
    out_list_reco_images = draw_reco_images(**in_dict_draw_reco, \
                                        in_font_scale=st.session_state.font_scale_sld, \
                                        in_conf_threshold=st.session_state.conf_threshold_sld)

    with show_reco.container():
        out_reco_columns = st.columns(in_nb_col, gap='medium')
        out_column_width = 400
        for ind_reco_col, reco_col in enumerate(out_reco_columns):
            if reader_type_list[ind_reco_col] == 'Tesseract':
                out_column_title = '<p style="font-size: 20px;color:rgb(0,0,0); \
                                   ">Recognition with ' + \
                                   in_reader_type_list[ind_reco_col] + \
                                  '<br>(with its own detector)</br></p>'
            else:
                out_column_title = '<p style="font-size: 20px;color:rgb(0,0,0); \
                                   ">Recognition with ' + \
                                   in_reader_type_list[ind_reco_col]+ '</p>'

            reco_col.markdown(out_column_title, unsafe_allow_html=True)

            if st.session_state.list_reco_status[ind_reco_col] == 'OK':
                reco_col.image(out_list_reco_images[ind_reco_col], width=out_column_width, \
                          use_column_width=True)
            else:
                reco_col.write(st.session_state.list_reco_status[ind_reco_col], \
                               use_column_width=True)


###################################################################################################
##   MAIN
###################################################################################################

##----------- Initializations ---------------------------------------------------------------------
print("PID : ", os.getpid())

st.title("OCR solutions comparator")
st.markdown("##### *EasyOCR, PPOCR, MMOCR, Tesseract*")
#st.markdown("#### PID : " + str(os.getpid()))

# Initializations
with st.spinner("Initializations in progress ..."):
    reader_type_list, reader_type_dict, color, list_dict_lang, \
    cols_size, dict_back_colors, fig_colorscale = initializations()

##----------- Choose language & image -------------------------------------------------------------
st.markdown("#### Choose languages for the text recognition")
lang_col = st.columns(4)
easyocr_key_lang = lang_col[0].selectbox(reader_type_list[0]+" :", list_dict_lang[0].keys(), 26)
easyocr_lang = list_dict_lang[0][easyocr_key_lang]
ppocr_key_lang = lang_col[1].selectbox(reader_type_list[1]+" :", list_dict_lang[1].keys(), 22)
ppocr_lang = list_dict_lang[1][ppocr_key_lang]
mmocr_key_lang = lang_col[2].selectbox(reader_type_list[2]+" :", list_dict_lang[2].keys(), 0)
mmocr_lang = list_dict_lang[2][mmocr_key_lang]
tesserocr_key_lang = lang_col[3].selectbox(reader_type_list[3]+" :", list_dict_lang[3].keys(), 35)
tesserocr_lang = list_dict_lang[3][tesserocr_key_lang]

image_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

##----------- Process input image -----------------------------------------------------------------
if image_file is not None:
    image_path, image_orig, image_cv2 = load_image(image_file)
    list_images = [image_orig, image_cv2]

##----------- Form with original image & hyperparameters for detectors ----------------------------
    with st.form("form1"):
        col1, col2 = st.columns(2, gap="medium")
        col1.markdown("##### Original image")
        col1.image(list_images[0], width=500, use_column_width=True)
        col2.markdown("##### Hyperparameters values for detection")
        hyper_tabs = col2.expander("Choose hyperparameters values for each detecter:", \
                                   expanded=False)

        tabs = hyper_tabs.tabs(reader_type_list)
        with tabs[0]:
            t0_min_size = st.slider("min_size", 1, 20, 10, step=1, \
                    help="min_size (int, default = 10) - Filter text box smaller than \
                          minimum value in pixel")
            t0_text_threshold = st.slider("text_threshold", 0.1, 1., 0.7, step=0.1, \
                    help="text_threshold (float, default = 0.7) - Text confidence threshold")
            t0_low_text = st.slider("low_text", 0.1, 1., 0.4, step=0.1, \
                    help="low_text (float, default = 0.4) - Text low-bound score")
            t0_link_threshold = st.slider("link_threshold", 0.1, 1., 0.4, step=0.1, \
                    help="link_threshold (float, default = 0.4) - Link confidence threshold")
            t0_canvas_size = st.slider("canvas_size", 2000, 5000, 2560, step=10, \
                    help='''canvas_size (int, default = 2560) \n
Maximum e size. Image bigger than this value will be resized down''')
            t0_mag_ratio = st.slider("mag_ratio", 0.1, 5., 1., step=0.1, \
                    help="mag_ratio (float, default = 1) - Image magnification ratio")
            t0_slope_ths = st.slider("slope_ths", 0.01, 1., 0.1, step=0.01, \
                    help='''slope_ths (float, default = 0.1) - Maximum slope \
                            (delta y/delta x) to considered merging. \n
Low valuans tiled boxes will not be merged.''')
            t0_ycenter_ths = st.slider("ycenter_ths", 0.1, 1., 0.5, step=0.1, \
                    help='''ycenter_ths (float, default = 0.5) - Maximum shift in y direction. \n
Boxes wiifferent level should not be merged.''')
            t0_height_ths = st.slider("height_ths", 0.1, 1., 0.5, step=0.1, \
                    help='''height_ths (float, default = 0.5) - Maximum different in box height. \n
Boxes wiery different text size should not be merged.''')
            t0_width_ths = st.slider("width_ths", 0.1, 1., 0.5, step=0.1, \
                    help="width_ths (float, default = 0.5) - Maximum horizontal \
                          distance to merge boxes.")
            t0_add_margin = st.slider("add_margin", 0.1, 1., 0.1, step=0.1, \
                    help='''add_margin (float, default = 0.1) - \
                            Extend bounding boxes in all direction by certain value. \n
This is rtant for language with complex script (E.g. Thai).''')
            t0_optimal_num_chars = st.slider("optimal_num_chars", None, 100, None, step=10, \
                    help="optimal_num_chars (int, default = None) - If specified, bounding boxes \
                          with estimated number of characters near this value are returned first.")

        with tabs[1]:
            t1_det_algorithm = st.selectbox('det_algorithm', ['DB'], \
            	    help='Type of detection algorithm selected. (default = DB)')
            t1_det_max_side_len = st.slider('det_max_side_len', 500, 2000, 960, step=10, \
                    help='''The maximum size of the long side of the image. (default = 960)\n
Limit thximum image height and width.\n
When theg side exceeds this value, the long side will be resized to this size, and the short side \
will be ed proportionally.''')
            t1_det_db_thresh =  st.slider('det_db_thresh', 0.1, 1., 0.3, step=0.1, \
                    help='''Binarization threshold value of DB output map. (default = 0.3) \n
Used to er the binarized image of DB prediction, setting 0.-0.3 has no obvious effect on the result.''')
            t1_det_db_box_thresh = st.slider('det_db_box_thresh', 0.1, 1., 0.6, step=0.1, \
                    help='''The threshold value of the DB output box. (default = 0.6) \n
DB post-essing filter box threshold, if there is a missing box detected, it can be reduced as appropriate. \n
Boxes sclower than this value will be discard.''')
            t1_det_db_unclip_ratio = st.slider('det_db_unclip_ratio', 1., 3.0, 1.6, step=0.1, \
                    help='''The expanded ratio of DB output box. (default = 1.6) \n
Indicatee compactness of the text box, the smaller the value, the closer the text box to the text.''')
            t1_det_east_score_thresh = st.slider('det_east_cover_thresh', 0.1, 1., 0.8, step=0.1, \
                    help="Binarization threshold value of EAST output map. (default = 0.8)")
            t1_det_east_cover_thresh = st.slider('det_east_cover_thresh', 0.1, 1., 0.1, step=0.1, \
                    help='''The threshold value of the EAST output box. (default = 0.1) \n
Boxes sclower than this value will be discarded.''')
            t1_det_east_nms_thresh = st.slider('det_east_nms_thresh', 0.1, 1., 0.2, step=0.1, \
                    help="The NMS threshold value of EAST model output box. (default = 0.2)")
            t1_det_db_score_mode = st.selectbox('det_db_score_mode', ['fast', 'slow'], \
                    help='''slow: use polygon box to calculate bbox score, fast: use rectangle box \
                    to calculate. (default = fast) \n
Use rectlar box to calculate faster, and polygonal box more accurate for curved text area.''')

        with tabs[2]:
            t2_det = st.selectbox('det', ['DB_r18','DB_r50','DBPP_r50','DRRG','FCE_IC15', \
                                          'FCE_CTW_DCNv2','MaskRCNN_CTW','MaskRCNN_IC15', \
                                          'MaskRCNN_IC17', 'PANet_CTW','PANet_IC15','PS_CTW',\
                                          'PS_IC15','Tesseract','TextSnake'], 10, \
            		help='Text detection algorithm. (default = PANet_IC15)')
            st.write("###### *More about text detection models*  👉  \
                    [here](https://mmocr.readthedocs.io/en/latest/textdet_models.html)")
            t2_merge_xdist = st.slider('merge_xdist', 1, 50, 20, step=1, \
                    help='The maximum x-axis distance to merge boxes. (defaut=20)')

        with tabs[3]:
            t3_psm = st.selectbox('Page segmentation mode (psm)', \
                                  [' -  Default', \
                                   ' 4  Assume a single column of text of variable sizes', \
                                   ' 5  Assume a single uniform block of vertically aligned text', \
                                   ' 6  Assume a single uniform block of text', \
                                   ' 7  Treat the image as a single text line', \
                                   ' 8  Treat the image as a single word', \
                                   ' 9  Treat the image as a single word in a circle', \
                                   '10  Treat the image as a single character', \
                                   '11  Sparse text. Find as much text as possible in no \
                                        particular order', \
                                   '13  Raw line. Treat the image as a single text line, \
                                        bypassing hacks that are Tesseract-specific'])
            t3_oem = st.selectbox('OCR engine mode', ['0  Legacy engine only', \
                                  '1  Neural nets LSTM engine only', \
                                  '2  Legacy + LSTM engines', \
                                  '3  Default, based on what is available'], 3)
            t3_whitelist = st.text_input('Limit tesseract to recognize only this characters :', \
                    placeholder='Limit tesseract to recognize only this characters', \
                    help='Example for numbers only : 0123456789')


        submit_detect = st.form_submit_button("Launch detection")

##----------- Process text detection --------------------------------------------------------------
    if submit_detect:
        # Process text detection

        if t0_optimal_num_chars == 0:
            t0_optimal_num_chars = None

        # Construct the config Tesseract parameter
        t3_config = ''
        psm = t3_psm[:2]
        if psm != ' -':
            t3_config += '--psm ' + psm.strip()
        oem = t3_oem[:1]
        if oem != '3':
            t3_config += ' --oem ' + oem
        if t3_whitelist != '':
            t3_config += ' -c tessedit_char_whitelist=' + t3_whitelist

        list_params_det = \
            [[easyocr_lang, \
              {'min_size': t0_min_size, 'text_threshold': t0_text_threshold, \
               'low_text': t0_low_text, 'link_threshold': t0_link_threshold, \
               'canvas_size': t0_canvas_size, 'mag_ratio': t0_mag_ratio, \
               'slope_ths': t0_slope_ths, 'ycenter_ths': t0_ycenter_ths, \
               'height_ths': t0_height_ths, 'width_ths': t0_width_ths, \
               'add_margin': t0_add_margin, 'optimal_num_chars': t0_optimal_num_chars \
              }], \
             [ppocr_lang, \
              {'det_algorithm': t1_det_algorithm, 'det_max_side_len': t1_det_max_side_len, \
               'det_db_thresh': t1_det_db_thresh, 'det_db_box_thresh': t1_det_db_box_thresh, \
               'det_db_unclip_ratio': t1_det_db_unclip_ratio, \
               'det_east_score_thresh': t1_det_east_score_thresh, \
               'det_east_cover_thresh': t1_det_east_cover_thresh, \
               'det_east_nms_thresh': t1_det_east_nms_thresh, \
               'det_db_score_mode': t1_det_db_score_mode}],
             [mmocr_lang, {'det': t2_det, 'merge_xdist': t2_merge_xdist}],
             [tesserocr_lang, {'lang': tesserocr_lang, 'config': t3_config}]
            ]

        show_info1 = st.empty()
        show_info1.info("Readers initializations in progress (it may take a while) ...")
        list_readers = init_readers(list_params_det)

        show_info1.info("Text detection in progress ...")
        list_images, list_coordinates = process_detect(image_path, list_images, list_readers, \
                                                       list_params_det, color)
        show_info1.empty()

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

##----------- Text detection results --------------------------------------------------------------
        st.subheader("Text detection")
        show_detect = st.empty()
        list_ok_detect = []
        with show_detect.container():
            columns = st.columns(st.session_state.columns_size, gap='medium')
            for no_col, col in enumerate(columns):
                column_title = '<p style="font-size: 20px;color:' + \
                               st.session_state.columns_color[no_col] + \
                               ';">Detection with ' + reader_type_list[no_col]+ '</p>'
                col.markdown(column_title, unsafe_allow_html=True)
                if isinstance(list_images[no_col+2], PIL.Image.Image):
                    col.image(list_images[no_col+2], width=st.session_state.column_width[no_col], \
                              use_column_width=True)
                    list_ok_detect.append(reader_type_list[no_col])
                else:
                    col.write(list_images[no_col+2], use_column_width=True)

        st.subheader("Text recognition")

##----------- Form with detection results & hyperparameters for recognition -----------------------
        with st.form("form2"):
            col1, col2 = st.columns([1,2])
            col1.markdown("##### Using detection performed above by:")
            col1.radio('Choose the detecter:', list_ok_detect, key='detect_reader', \
                                               horizontal=False)
            col2.markdown("##### Hyperparameters values for recognition")
            hyper_tabs = col2.expander("Choose hyperparameters values for each detecter:", \
                                       expanded=False)

            tabs = hyper_tabs.tabs(reader_type_list)
            with tabs[0]:
                t0_decoder = st.selectbox('decoder', ['greedy', 'beamsearch', 'wordbeamsearch'], \
                    help="decoder (string, default = 'greedy') - options are 'greedy', \
                         'beamsearch' and 'wordbeamsearch.")
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
        Usefor specific problem (E.g. license plate, etc.)''')
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
        Firs with original image and second with contrast adjusted to 'adjust_contrast' value.\n
        The with more confident level will be returned as a result.''')
                t0_adjust_contrast = st.slider('adjust_contrast', 0.1, 1., 0.5, step=0.1, \
                    help = 'adjust_contrast (float, default = 0.5) - target contrast level for low \
                    contrast text box')

            with tabs[1]:
                t1_rec_algorithm = st.selectbox('rec_algorithm', ['CRNN', 'SVTR_LCNet'], 0, \
                    help="Type of recognition algorithm selected. (default=CRNN)")
                t1_rec_batch_num = st.slider('rec_batch_num', 1, 50, step=1, \
                    help="When performing recognition, the batchsize of forward images. \
                         (default=30)")
                t1_max_text_length = st.slider('max_text_length', 3, 250, 25, step=1, \
                    help="The maximum text length that the recognition algorithm can recognize. \
                         (default=25)")
                t1_use_space_char = st.radio('use_space_char', [True, False], 0, horizontal=True, \
                    help="Whether to recognize spaces. (default=TRUE)")
                t1_drop_score = st.slider('drop_score', 0., 1., 0.25, step=.05, \
                    help="Filter the output by score (from the recognition model), and those \
                          below this score will not be returned. (default=0.5)")

            with tabs[2]:
                t2_recog = st.selectbox('recog', ['ABINet','CRNN','CRNN_TPS','MASTER', \
                              'NRTR_1/16-1/8','NRTR_1/8-1/4','RobustScanner','SAR','SAR_CN', \
                              'SATRN','SATRN_sm','SEG','Tesseract'], 7, \
                        help='Text recognition algorithm. (default = SAR)')
                st.write("###### *More about text recognition models*  👉  \
                        [here](https://mmocr.readthedocs.io/en/latest/textrecog_models.html)")

            with tabs[3]:
                t3r_psm = st.selectbox('Page segmentation mode (psm)', \
                                       [' -  Default', \
                                       ' 4  Assume a single column of text of variable sizes', \
                                       ' 5  Assume a single uniform block of vertically aligned \
                                            text', \
                                       ' 6  Assume a single uniform block of text', \
                                       ' 7  Treat the image as a single text line', \
                                       ' 8  Treat the image as a single word', \
                                       ' 9  Treat the image as a single word in a circle', \
                                       '10  Treat the image as a single character', \
                                       '11  Sparse text. Find as much text as possible in no \
                                            particular order', \
                                       '13  Raw line. Treat the image as a single text line, \
                                            bypassing hacks that are Tesseract-specific'])
                t3r_oem = st.selectbox('OCR engine mode', ['0  Legacy engine only', \
                                       '1  Neural nets LSTM engine only', \
                                       '2  Legacy + LSTM engines', \
                                       '3  Default, based on what is available'], 3)
                t3r_whitelist = st.text_input('Limit tesseract to recognize only this \
                                    characters :', placeholder='Limit tesseract to recognize only \
                                    this characters', help='Example for numbers only : 0123456789')

            submit_reco = st.form_submit_button("Launch recognition")

        if submit_reco:
            process_detect.clear()
##----------- Hightlight the detecter --------------------------------------
            show_detect.empty()
            with show_detect.container():
                columns_size = [1 for x in reader_type_list]
                column_width  = [400 for x in reader_type_list]
                columns_color = ["rgb(0,0,0)" for x in reader_type_list]
                columns_size[reader_type_dict[st.session_state.detect_reader]] = 2
                column_width[reader_type_dict[st.session_state.detect_reader]] = 500
                columns_color[reader_type_dict[st.session_state.detect_reader]] = "rgb(228,26,28)"
                columns = st.columns(columns_size, gap='medium')

                for ind_col, col in enumerate(columns):
                    column_title = '<p style="font-size: 20px;color:'+columns_color[ind_col] + \
                                    ';">Detection with ' + reader_type_list[ind_col]+ '</p>'
                    col.markdown(column_title, unsafe_allow_html=True)
                    if isinstance(list_images[ind_col+2], PIL.Image.Image):
                        col.image(list_images[ind_col+2], width=column_width[ind_col], \
                                  use_column_width=True)
                    else:
                        col.write(list_images[ind_col+2], use_column_width=True)
                st.session_state.columns_size = columns_size
                st.session_state.column_width = column_width
                st.session_state.columns_color = columns_color

##----------- Process recognition ------------------------------------------
            reader_ind = reader_type_dict[st.session_state.detect_reader]
            list_boxes = list_coordinates[reader_ind]

            # Construct the config Tesseract parameter
            t3r_config = ''
            psm = t3r_psm[:2]
            if psm != ' -':
                t3r_config += '--psm ' + psm.strip()
            oem = t3r_oem[:1]
            if oem != '3':
                t3r_config += ' --oem ' + oem
            if t3r_whitelist != '':
                t3r_config += ' -c tessedit_char_whitelist=' + t3r_whitelist

            list_params_rec = \
                [{'decoder': t0_decoder, 'beamWidth': t0_beamWidth, \
                  'batch_size': t0_batch_size, 'workers': t0_workers, \
                  'allowlist': t0_allowlist, 'blocklist': t0_blocklist, \
                  'detail': t0_detail, 'paragraph': t0_paragraph, \
                  'contrast_ths': t0_contrast_ths, 'adjust_contrast': t0_adjust_contrast
                 },
                 { **list_params_det[1][1], **{'rec_algorithm': t1_rec_algorithm, \
                 'rec_batch_num': t1_rec_batch_num, 'max_text_length': t1_max_text_length, \
                  'use_space_char': t1_use_space_char, 'drop_score': t1_drop_score}, \
                  **{'lang': list_params_det[1][0]}
                 },
                 {'recog': t2_recog},
                 {'lang': tesserocr_lang, 'config': t3r_config}
                ]

            show_info2 = st.empty()

            with show_info2.container():
                st.info("Text recognition in progress ...")
                df_results, list_reco_images, df_results_tesseract, list_reco_status = \
                    process_recog(list_readers, list_images[1], list_boxes, \
                                  list_params_rec, dict_back_colors)
            show_info2.empty()

            st.session_state.df_results = df_results
            st.session_state.list_reco_images = list_reco_images
            st.session_state.list_boxes = list_boxes
            st.session_state.df_results_tesseract = df_results_tesseract
            st.session_state.list_reco_status = list_reco_status

        if 'df_results' in st.session_state:
##----------- Show recognition results ------------------------------------------------------------
            results_cols = st.session_state.df_results.columns
            list_col_text = np.arange(1, len(cols_size), 2)
            list_col_confid = np.arange(2, len(cols_size), 2)

            dict_draw_reco = {'in_image': st.session_state.list_images[1], \
                              'in_boxes_coordinates': st.session_state.list_boxes, \
                              'in_list_texts': [st.session_state.df_results[x].to_list() \
                                                for x in results_cols[list_col_text]], \
                              'in_list_confid': [st.session_state.df_results[x].to_list() \
                                                for x in results_cols[list_col_confid]], \
                              'in_dict_back_colors': dict_back_colors,
                              'in_df_results_tesseract' : st.session_state.df_results_tesseract
                              }
            show_reco = st.empty()

            with st.form("form3"):
                st.plotly_chart(fig_colorscale, use_container_width=True)

                col_font, col_threshold = st.columns(2)

                col_font.slider('Font scale', 1, 7, 4, step=1, key="font_scale_sld")
                col_threshold.slider('% confidence threshold for text color change', 40, 100, 64, \
                                    step=1, key="conf_threshold_sld")
                col_threshold.write("(text color is black below this % confidence threshold, \
                                     and white above)")

                with show_reco.container():
                    reco_columns = st.columns(len(reader_type_list), gap='medium')
                    column_width = 400
                    for ind_col, col in enumerate(reco_columns):
                        if reader_type_list[ind_col] == 'Tesseract':
                            column_title = '<p style="font-size: 20px;color:rgb(0,0,0); \
                                            ">Recognition with ' + \
                                            reader_type_list[ind_col] + \
                                            '<br>(with its own detector)</br></p>'
                        else:
                            column_title = '<p style="font-size: 20px;color:rgb(0,0,0); \
                                            ">Recognition with ' + \
                                            reader_type_list[ind_col]+ '</p>'
                        col.markdown(column_title, unsafe_allow_html=True)
                        if st.session_state.list_reco_status[ind_col] == 'OK':
                            col.image(st.session_state.list_reco_images[ind_col], \
                                      width=column_width, use_column_width=True)
                        else:
                            col.write(st.session_state.list_reco_status[ind_col], \
                                      use_column_width=True)

                submit_resize = st.form_submit_button("Refresh")

            if submit_resize:
                update_font_scale(len(reader_type_list), dict_draw_reco, reader_type_list)

            st.subheader("Recognition details")
            with st.expander("Detailed areas", expanded=False):

                tab_ocr, tab_tesseract = st.tabs(["EasyOCR, PPOCR, MMOCR", "Tesseract"])

                with tab_ocr:
                    cols = st.columns(cols_size)
                    cols[0].markdown('#### Detected area')
                    for i in range(1, (len(reader_type_list)-1)*2, 2):
                        cols[i].markdown('#### with ' + reader_type_list[i//2])

                    for row in st.session_state.df_results.itertuples():
                        #cols = st.columns(1 + len(reader_type_list)*2)
                        cols = st.columns(cols_size)
                        cols[0].image(row.cropped_image, width=150)
                        for ind_col in range(1, len(cols), 2):
                            cols[ind_col].write(getattr(row, results_cols[ind_col]))
                            cols[ind_col+1].write("("+str( \
                                getattr(row, results_cols[ind_col+1]))+"%)")

                    st.download_button(
                        label="Download results as CSV file",
                        data=convert_df(st.session_state.df_results),
                        file_name='OCR_comparator_results.csv',
                        mime='text/csv',
                    )

                if not st.session_state.df_results_tesseract.empty:
                    with tab_tesseract:
                        cols = st.columns([2,2,1])
                        cols[0].markdown('#### Detected area')
                        cols[1].markdown('#### with Tesseract')

                        for row in st.session_state.df_results_tesseract.itertuples():
                            cols = st.columns([2,2,1])
                            cols[0].image(row.cropped, width=150)
                            cols[1].write(getattr(row, 'text'))
                            cols[2].write("("+str(getattr(row, 'conf'))+"%)")

                        st.download_button(
                            label="Download Tesseract results as CSV file",
                            data=convert_df(st.session_state.df_results),
                            file_name='OCR_comparator_Tesseract_results.csv',
                            mime='text/csv',
                        )