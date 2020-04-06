import argparse
import os
import shutil
import random
from tqdm import tqdm
from xml.dom import minidom


parser = argparse.ArgumentParser(prog='test.py')
parser.add_argument('--dir', type=str, help='raw data path')
parser.add_argument('--split', type=float, default=0.2, help='validation split')
args = parser.parse_args()

assert os.path.isdir(args.dir), Exception(f"{args.dir} can not be seen from {os.getcwd()}")

cooked = args.dir + '_cooked'

for folder in [cooked, cooked + '/images', cooked + '/labels']:
    if not os.path.isdir(folder):
        os.mkdir(folder)

names = {}
images = []
labels = []

files = os.listdir(args.dir)
for file in tqdm(files):
    f = args.dir+'/'+file
    if file.endswith("xml"):
        root = minidom.parse(f)
        sizes = root.getElementsByTagName("size")
        try:
            size = {el.tagName: int(el.firstChild.data) for el in root.getElementsByTagName("size")[0].childNodes if not isinstance(el, minidom.Text)}
        except:
            print()
        domobjects = root.getElementsByTagName("object")
        boxes = []
        for domobject in domobjects:
            name = domobject.getElementsByTagName("name")[0].firstChild.data
            if name not in names:
                names[name] = len(names)

            bndbox = domobject.getElementsByTagName("bndbox")[0].childNodes

            xyxy = {el.tagName: int(el.firstChild.data) for el in bndbox if not isinstance(el, minidom.Text)}
            xmin, ymin, xmax, ymax = xyxy["xmin"], xyxy["ymin"], xyxy["xmax"], xyxy["ymax"]
            if not size['width'] or not size['height']:
                continue
            x = ((xmax + xmin) / 2) / size["width"]
            y = ((ymax + ymin) / 2) / size["height"]
            w = (xmax - xmin) / size["width"]
            h = (ymax - ymin) / size["height"]
            boxes.append(f"{names[name]} {x} {y} {w} {h}")
        with open(cooked+'/labels/'+file.replace("xml", "txt"), "w") as f:
            f.write('\n'.join(boxes))

    else:
        shutil.copy(f, cooked+'/images/'+file)
        images.append(cooked+'/images/'+file)

random.shuffle(images)

with open(cooked+'/train.txt', "w") as f:
    f.write('\n'.join(images[:int(len(images) * (1 - args.split))]))

with open(cooked+'/valid.txt', "w") as f:
    f.write('\n'.join(images[int(len(images) * (1 - args.split)):]))

with open(cooked+'/classes.names', "w") as f:
    f.write("\n".join(c for c in names))

with open(cooked+'/data.txt', "w") as f:
    f.write(f"classes={len(names)}\ntrain={cooked+'/train.txt'}\nvalid={cooked+'/valid.txt'}\nnames={cooked+'/classes.names'}")