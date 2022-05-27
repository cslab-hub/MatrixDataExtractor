import os
import numpy as np
from secrets import choice
import numpy as np
import streamlit as st
import cv2 as cv
from operator import itemgetter
from itertools import groupby
import pytesseract
import pandas as pd


# Resize grayscale image
def image_gray_resize(img, width=None,height=None, inter = cv.INTER_AREA):
    
    dim = (width, height)
    img = cv.resize(img,dim,interpolation=inter)
    return img

# Store row separator pixel values in list to draw lines on image
def get_row_col_separator_list(img):
    img_height, img_width = img.shape

    white_row_width_list = list()
    temp_white_row_width_list = list()
    white_col_width_list = list()
    temp_white_col_width_list = list()
    draw_row_separator_list = list()
    draw_col_separator_list = list()
    

    # Create row separator list  
    for row in range(img_height):
        # Check each image col is'black or not
        if np.all(img[row,:] == 255): # black col = 0, white col = 255
            temp_white_row_width_list.append(row)

    #ref: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
    for k, g in groupby(enumerate(temp_white_row_width_list), lambda i_x: i_x[0] - i_x[1]):
        temp_white_row_width_list = list(map(itemgetter(1), g))
        #print(sorted(temp_white_row_width_list))
        white_row_width_list.append(temp_white_row_width_list)


    for inner_list in white_row_width_list:
        middle_index = int((len(inner_list) - 1)/2)
        draw_point = inner_list[middle_index]
        draw_row_separator_list.append(draw_point)

    # Create col separator list  
    for col in range(img_width):
        # Check each image col is'black or not
        if np.all(img[:,col] == 255): # black col = 0, white col = 255
            temp_white_col_width_list.append(col)

    #ref: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
    for k, g in groupby(enumerate(temp_white_col_width_list), lambda i_x: i_x[0] - i_x[1]):
        temp_white_col_width_list = list(map(itemgetter(1), g))
        white_col_width_list.append(temp_white_col_width_list)


    for inner_list in white_col_width_list:
        middle_index = int((len(inner_list) - 1)/2)
        draw_point = inner_list[middle_index]
        draw_col_separator_list.append(draw_point)
    
    return draw_row_separator_list, draw_col_separator_list

def draw_row_separator(img, draw_row_separator_list, user_del_rowpixel_list):
    img_height, img_width = img.shape
    if len(draw_row_separator_list) !=0:
        if len(user_del_rowpixel_list) !=0:
            temp_row_separator_list = [x for x in draw_row_separator_list if x not in user_del_rowpixel_list]
            for draw_point in temp_row_separator_list:
                cv.line(img,(0,draw_point),(img_width,draw_point),(0,0,0),2)
        else:
            for draw_point in draw_row_separator_list:
                cv.line(img,(0,draw_point),(img_width,draw_point),(0,0,0),2)
    return img

def draw_col_separator(img, draw_col_separator_list, user_del_colpixel_list):
    img_height, img_width = img.shape
    if len(draw_col_separator_list) !=0:
        if len(user_del_colpixel_list) !=0:
            temp_col_separator_list = [x for x in draw_col_separator_list if x not in user_del_colpixel_list]
            for draw_point in temp_col_separator_list:
                cv.line(img,(draw_point,0),(draw_point,img_height),(0,0,0),2)
        else:
            for draw_point in draw_col_separator_list:
                cv.line(img,(draw_point,0),(draw_point,img_height),(0,0,0),2)
    return img

# For semi-border-row table
def draw_semi_bor_col_separator(img, temp_col_separator_list):
    img_height, img_width = img.shape
    if len(temp_col_separator_list) !=0:
            for draw_point in temp_col_separator_list:
                cv.line(img,(draw_point,0),(draw_point,img_height),(0,0,0),2)

    return img

# For semi-border-col table
def draw_semi_bor_row_separator(img, temp_row_separator_list):
    img_height, img_width = img.shape
    if len(temp_row_separator_list) !=0:
        for draw_point in temp_row_separator_list:
            cv.line(img,(0,draw_point),(img_width,draw_point),(0,0,0),2)
    return img

# Save images after drawing
def save_draw_img(img, img_filename):
    save_img_file = "util/data/tabstructimg/" + img_filename
    cv.imwrite(save_img_file, img)

# Fully bordered Table detection
def detect_table_structure(img, img_file_woext, ver_ker_len_param=128, hor_ker_len_param=32, iter_no=2, ver_strcelem=5,hor_strcelem=5):
    #img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_height, img_width = img.shape
    (thresh, img_bin) = cv.threshold(img, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    img_bin = cv.bitwise_not(img_bin)

    # Vertical Kernel
    kernel_length_ver = (np.array(img).shape[1])//ver_ker_len_param # Semi-bor=128 or Un-bor=64 with 1024 x 512
    ver_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_length_ver)) 
    ver_eroded_img = cv.erode(img_bin, ver_kernel, iterations=iter_no)
    img_ver_lines = cv.dilate(ver_eroded_img, ver_kernel, iterations=iter_no)

    # Horizontal Kernel
    kernel_length_hor = (np.array(img).shape[1])//hor_ker_len_param
    hor_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length_hor, 1))
    hor_eroded_img = cv.erode(img_bin, hor_kernel, iterations=iter_no)
    img_hor_lines = cv.dilate(hor_eroded_img, hor_kernel, iterations=iter_no)

    # Get Table Structure
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (hor_strcelem, ver_strcelem)) #anchor=(-1, -1)
    table_segment = cv.addWeighted(img_ver_lines, 0.5, img_hor_lines, 0.5, 0.0)
    table_segment = cv.erode(cv.bitwise_not(table_segment), kernel, iterations=iter_no)
    thresh, table_segment = cv.threshold(table_segment, 0, 255, cv.THRESH_OTSU)

    # Detect Contours
    contours, hierarchy = cv.findContours(table_segment, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv.boundingRect(contour) for contour in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x:x[1][1]))
    boxes = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if (w<1024 and h<512):
            image = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            boxes.append([x,y,w,h])

    # Store Fully bordered Table values in row-column format
    rows=[]
    columns=[]
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    columns.append(boxes[0])
    previous=boxes[0]
    for i in range(1,len(boxes)):
        if(boxes[i][1]<=previous[1]+mean/2):
            columns.append(boxes[i])
            previous=boxes[i]
            if(i==len(boxes)-1):
                rows.append(columns)
        else:
            rows.append(columns)
            columns=[]
            previous = boxes[i]
            columns.append(boxes[i])

    row_no = len(rows)
    col_no = int(len(boxes)/len(rows))

    #calculating maximum number of cells
    countcol = 0
    for i in range(len(rows)):
        countcol = len(rows[i])
        if countcol > countcol:
            countcol = countcol

    #Retrieving the center of each column
    center = [int(rows[i][j][0]+rows[i][j][2]/2) for j in range(len(rows[i])) if rows[0]]
    center=np.array(center)
    center.sort()
        
    #Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxes = []
    for i in range(len(rows)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(rows[i])):
            diff = abs(center-(rows[i][j][0]+rows[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(rows[i][j])
        finalboxes.append(lis)

    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=""
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    finalimg = img[x:x+h, y:y+w]
                    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
                    border = cv.copyMakeBorder(finalimg,2,2,2,2,   cv.BORDER_CONSTANT,value=[255,255])
                    resizing = cv.resize(border, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
                    dilation = cv.dilate(resizing, kernel,iterations=1)
                    erosion = cv.erode(dilation, kernel,iterations=1)

                    
                    out = pytesseract.image_to_string(erosion)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner +" "+ out
                    inner = inner.replace('\x0c', '').replace('\\x0c', '').replace('\n', '').replace('?', '')
                outer.append(inner)

    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(rows),countcol))
    data = dataframe.style.set_properties(align="left")
    save_excel_file = "util/data/tableinfo/" + img_file_woext + ".xlsx"
    data.to_excel(save_excel_file)
    
    return img, row_no, col_no, dataframe


# Main Function of Streamlit Application
def main():
    st.title("Know Your Table Structure")
    st.sidebar.title("Table Image")
    st.sidebar.subheader("Parameters")
    #selection_option = 
    selection_option = ['Draw and Save Table Structure', 'Verify Table Structure'] 

    choice = st.sidebar.radio('Choose the app mode', selection_option)

    img_width = 2048
    img_height = 1024

    if choice == 'Draw and Save Table Structure':
        st.subheader('Draw border on Table Image and save')
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        # https://github.com/streamlit/streamlit/issues/888
        if image_file is not None:
            img_filename = image_file.name
            imgfile_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv.imdecode(imgfile_bytes, cv.IMREAD_GRAYSCALE)
            img = image_gray_resize(img,width=img_width,height=img_height)
            st.image(img)

        st.subheader('Process and Save Table Image of 2048 x 1024 (W x H)')
        table_types = ['Unbordered Table','Semi-bordered Table']
        choice = st.sidebar.selectbox('Choose Your Table', table_types)

        
        if choice == 'Unbordered Table':
            if st.button("Process Unbordered Table"):
                # Row and Col separator list
                draw_row_separator_list, draw_col_separator_list = get_row_col_separator_list(img)
                st.text("Row Separator value: "+ str(draw_row_separator_list))
                st.text("Column Separator value: "+str(draw_col_separator_list))
                st.image(img)

            unbor_row_form = st.form(key='unbor_row_form')
            row_sep_list = unbor_row_form.text_input(label='Enter your row-pixel values using comma separator:')
            col_sep_list = unbor_row_form.text_input(label='Enter your col-pixel values using comma separator:')
            draw_unbortable_button = unbor_row_form.form_submit_button(label='Draw Table')
            st.write('Press Draw Table button and then Save Image Checkbox')
            
            save_img = st.checkbox('Save Image')
            if draw_unbortable_button:
                if len(col_sep_list) !=0 and len(row_sep_list) ==0:
                    temp_col_list = col_sep_list.split(",")
                    temp_col_separator_list = [int(x) for x in temp_col_list]
                    img = draw_semi_bor_col_separator(img,temp_col_separator_list)
                    st.image(img)

                    if save_img:
                        save_draw_img(img, img_filename)


                elif len(row_sep_list) !=0 and len(col_sep_list) ==0 :
                    temp_row_list = row_sep_list.split(",")
                    temp_row_separator_list = [int(x) for x in temp_row_list]
                    img = draw_semi_bor_row_separator(img,temp_row_separator_list)
                    st.image(img)

                    if save_img:
                        save_draw_img(img, img_filename)

                elif len(col_sep_list) !=0 and len(row_sep_list) !=0:
                    temp_row_list = row_sep_list.split(",")
                    temp_col_list = col_sep_list.split(",")
                    temp_row_separator_list = [int(x) for x in temp_row_list]
                    temp_col_separator_list = [int(x) for x in temp_col_list]
                    img = draw_semi_bor_row_separator(img,temp_row_separator_list)
                    img = draw_semi_bor_col_separator(img,temp_col_separator_list)
                    st.image(img)

                    if save_img:
                        save_draw_img(img, img_filename)    



        if choice == 'Semi-bordered Table':
            semi_bor_form = st.form(key='semi_bor_form')
            row_sep_list = semi_bor_form.text_input(label='Enter your row-pixel values using comma separator:')
            col_sep_list = semi_bor_form.text_input(label='Enter your col-pixel values using comma separator:')
            draw_table_button = semi_bor_form.form_submit_button(label='Draw Table')
            st.write('Press Draw Table and then Save Table button')
            
            save_img = st.checkbox('Save Image')
            
            if draw_table_button:
                if len(col_sep_list) !=0 and len(row_sep_list) ==0:
                    temp_col_list = col_sep_list.split(",")
                    temp_col_separator_list = [int(x) for x in temp_col_list]
                    img = draw_semi_bor_col_separator(img,temp_col_separator_list)
                    st.image(img)

                    if save_img:
                        save_draw_img(img, img_filename)


                elif len(row_sep_list) !=0 and len(col_sep_list) ==0 :
                    temp_row_list = row_sep_list.split(",")
                    temp_row_separator_list = [int(x) for x in temp_row_list]
                    img = draw_semi_bor_row_separator(img,temp_row_separator_list)
                    st.image(img)

                    if save_img:
                        save_draw_img(img, img_filename)

                elif len(col_sep_list) !=0 and len(row_sep_list) !=0:
                    temp_row_list = row_sep_list.split(",")
                    temp_col_list = col_sep_list.split(",")
                    temp_row_separator_list = [int(x) for x in temp_row_list]
                    temp_col_separator_list = [int(x) for x in temp_col_list]
                    img = draw_semi_bor_row_separator(img,temp_row_separator_list)
                    img = draw_semi_bor_col_separator(img,temp_col_separator_list)
                    st.image(img)

                    if save_img:
                        save_draw_img(img, img_filename)

                else:
                    st.image(img)
                
                
    elif choice == 'Verify Table Structure':
        st.subheader('Verify Bordered Table Structure')
        image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
        # https://github.com/streamlit/streamlit/issues/888
        if image_file is not None:
            img_filename = image_file.name
            img_file_woext = os.path.splitext(os.path.basename(img_filename))[0]
            imgfile_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            img = cv.imdecode(imgfile_bytes, cv.IMREAD_GRAYSCALE)
            img = image_gray_resize(img,width=img_width,height=img_height)
            #img = detect_table_structure(img)
            #img =  detect_table_structure(img, ver_ker_len_param=128, hor_ker_len_param=32, iter_no=2, ver_strcelem=5,hor_strcelem=5)
            st.image(img)

        st.subheader('Process and Save Table Structure - Default W x H - 2048 x 1024')

        table_struct_form = st.form(key='table_struct_form')
        hor_ker_len_param = table_struct_form.text_input(label='Enter Horizontal Kernel Length Param:')
        ver_ker_len_param = table_struct_form.text_input(label='Enter Vertical Kernel Length Param:')
        iter_no = table_struct_form.text_input(label='Enter Iteration No.:')
        hor_strcelem = table_struct_form.text_input(label='Enter Horizontal GetStruct Kernel:')
        ver_strcelem = table_struct_form.text_input(label='Enter Vertical GetStruct Kernel:')
        resize_img_width = table_struct_form.text_input(label='Enter Img Resize Width:')
        resize_img_height = table_struct_form.text_input(label='Enter Img Resize Height.:')
        draw_table_struct_button = table_struct_form.form_submit_button(label='Verify Table Structure')
        st.write('Press Verify Table Structure button and then Save Table Structure Checkbox')

        save_tab_struct_img = st.checkbox('Resize')
        show_table_data = st.checkbox('Show Table Data')
        if draw_table_struct_button:
            if save_tab_struct_img:
                img = cv.resize(img, (int(resize_img_width), int(resize_img_height)))
                img, row_no, col_no, dataframe = detect_table_structure(img, img_file_woext, ver_ker_len_param=int(ver_ker_len_param), 
                                    hor_ker_len_param=int(hor_ker_len_param), iter_no=int(iter_no), ver_strcelem=int(ver_strcelem),hor_strcelem=int(hor_strcelem))
                st.write('Row: ', row_no, " , Col: ", col_no)                    
                #save_table_info(img, img_filename)
                st.image(img)
                
                if show_table_data:
                    st.write('Your Table Data')
                    st.write(dataframe)
                else:
                    pass
                
            else:
                img, row_no, col_no, dataframe = detect_table_structure(img, img_file_woext, ver_ker_len_param=int(ver_ker_len_param),
                                    hor_ker_len_param=int(hor_ker_len_param), iter_no=int(iter_no), ver_strcelem=int(ver_strcelem),hor_strcelem=int(hor_strcelem))
                st.write('Row: ', row_no, " , Col: ", col_no)
                st.image(img)
                
                if show_table_data:
                    st.write('Your Table Data')
                    st.write(dataframe)
                else:
                    pass
                

                                  
if __name__=='__main__':
    main()