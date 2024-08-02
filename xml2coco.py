import xml.etree.ElementTree as ET
import os
import random

def vocbox2cocobox(bbox,size):
    xmin, ymin, xmax, ymax = bbox
    x = (xmin+xmax)/2. - 1
    y = (ymin+ymax)/2. - 1
    ix = x/size[0]
    iy = y/size[1]
    w = xmax-xmin
    h = ymax-ymin
    iw = w/size[0]
    ih = h/size[1]
    return ix,iy,iw,ih
    pass

def load_xml(path,save_path):
    cls2id = {}
    save_file =  open(save_path,'w')
    tree = ET.parse(open(path))
    root = tree.getroot()
    size = root.find('size')
    h = int(size.find('height').text)
    w = int(size.find('width').text)
    content = ''
    for obj in root.iter('object'):
        cls = obj.find('name').text
        # if cls not in cls2id:
        #     continue
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        if xmax > w:
            xmax = w
        if ymax > h:
            ymax = h
        cocobbox = vocbox2cocobox((xmin,ymin,xmax,ymax),(h,w))
        result = [cls] + list(cocobbox)
        content += ' '.join([str(i) for i in result])+'\n'
    save_file.write(content)
    save_file.close()

def split(img_dir):
    random.seed(2024)
    # xml_dir  = '/home/aistudio/dataset/Annotations'#标签文件地址
    # img_dir = '/home/aistudio/dataset/JPEGImages'#图像文件地址
    path_list = list()
    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir,img)
        # xml_path = os.path.join(xml_dir,'.'.join(img.split('.')[:-1])+'.xml')
        path_list.append(img_path)
    random.shuffle(path_list)
    ratio = 0.8 #测试集和验证集划分比例0.8:0.2
    train_f = open('train.txt','w') #生成训练文件
    val_f = open('val.txt' ,'w')#生成验证文件

    for i ,content in enumerate(path_list):
        # img, xml = content
        img = content
        text = img + '\n'
        # text = img + ' ' + xml + '\n'
        if i < len(path_list) * ratio:
            train_f.write(text)
        else:
            val_f.write(text)
    train_f.close()
    val_f.close()

def generate_label(label):
    #生成标签文档
    # label = ['Roller','Grader','Loader','Excavator','Bulldozer','Mixer Truck','Mobile Crane','Dump Truck']#设置你想检测的类别
    with open('/home/aistudio/work/dataset_processed/label_list.txt', 'w') as f:
        for text in label:
             #x2coco的labels_str = f.read().split('')改为labels_str = f.read().split('\n')
            f.write(text+'\n')

if __name__ == '__main__':
    path = './xml/aqd_wpd_0002.xml'
    save_path = './txt/aqd_wpd_0002.txt'
    load_xml(path,save_path)