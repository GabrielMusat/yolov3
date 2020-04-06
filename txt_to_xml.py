import argparse
from typing import List
import os
import cv2
import shutil
from tqdm import tqdm


parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('--dir', type=str, help='.jpg and .txt files directory')
parser.add_argument('--images-dir', type=str, help='original images folder')
parser.add_argument('--output', type=str, help='name of output .xml files folder')
args = parser.parse_args()

output = args.output if args.output is not None else args.dir+'_reversed'
if not os.path.isdir(output):
    os.mkdir(output)


def make(filename: str, width: int, height: int, objects: List[dict]) -> str:
    text = f'<annotation>\n\t<filename>{filename}</filename>\n\t<size>\n\t\t<width>{width}</width>\n\t\t<height>{height}</height>\n\t</size>'
    for element in objects:
        text += f'\n\t<object>\n\t\t<name>{element["name"]}</name>\n\t\t<bndbox>\n\t\t\t<xmin>{element["xmin"]}</xmin>\n\t\t\t<ymin>{element["ymin"]}</ymin>\n\t\t\t<xmax>{element["xmax"]}</xmax>\n\t\t\t<ymax>{element["ymax"]}</ymax>\n\t\t</bndbox>\n\t</object>'
    text += '\n</annotation>'
    return text


for file in tqdm(os.listdir(args.dir)):
    filepath = args.dir+'/'+file
    if file.split('.')[-1] == 'txt':
        continue
    else:
        array = cv2.imread(filepath)
        width, height = array.shape[1], array.shape[0]

        mush_class = ' '.join((file.split('_')[:2])).lower()

        objects = []
        if os.path.isfile(filepath+'.txt'):
            with open(filepath+'.txt') as f:
                boxes = f.read().split('\n')

            for box in boxes:
                if len(box) <= 1:
                    continue
                coord = box.split(' ')
                objects.append({
                    "name": mush_class,
                    "xmin": coord[0],
                    "ymin": coord[1],
                    "xmax": coord[2],
                    "ymax": coord[3]
                })

            data = make(file, width, height, objects)
            with open(output+'/'+'.'.join(file.split('.')[:-1])+'.xml', 'w') as f:
                f.write(data)

        shutil.copy(args.images_dir+'/'+file, output+'/'+file)
