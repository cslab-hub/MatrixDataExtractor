import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def get_image_id(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        filename = filename.replace('_', 'dp')
        return filename
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        file_name = root.find('filename').text
        img_id = get_image_id(file_name)
        img_width = int(root.find('size')[0].text)
        img_height = int(root.find('size')[1].text)
                     

        for member in root.findall('object'):
            bbox = list()
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            w = int(xmax - xmin)
            h = int(ymax - ymin)
            label = member.find('name').text
            bbox = [xmin, ymin, w, h]
            area = int(w * h)
            

            value = (file_name, img_id, int(img_width), int(img_height), label,
                     bbox, xmin, ymin, w, h, xmax, ymax, area
                     )
            xml_list.append(value)

    column_name = ['file_name', 'image_id', 'width', 'height', 'labels', 
                   'bbox', 'xmin', 'ymin', 'bbox_width', 'bbox_height', 'xmax', 'ymax', 'area'
                    
                  ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
    

def main():
    datasets = ['train', 'val', 'test']
    for ds in datasets:
        img_folder = str(ds + '2021/images')
        image_path = os.path.join(os.getcwd(), img_folder)
        if(os.path.isdir(image_path)):
            print("Dir exists")
        else:
            print("Check your Dir")
        xml_df = xml_to_csv(image_path)
        csv_path = str('annotations')
        xml_df.to_csv(os.path.join(csv_path , 'diplast2021_{}.csv'.format(ds)), index=None)
        print('Successfully converted xml to csv.')


if __name__ == "__main__":
    main()
