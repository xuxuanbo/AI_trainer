#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp

import imgviz
import numpy as np

import labelme
from PIL import Image
import os
import cv2
import numpy as np



def visualization(original_path,mask_path,save_dir):
    file_name = '.'.join(os.path.basename(original_path).split('.')[:-1])
    # 打开图像文件
    img = Image.open(original_path)
    # 转换为RGB模式（如果需要）
    img = img.convert('RGB')
    img_array = np.array(img)
    img_gray_array = imgviz.rgb2gray(img_array)
    mask = Image.open(mask_path)
    # 转换为RGB模式（如果需要）
    # mask = img.convert('RGB')
    mask_array = np.array(mask)
    viz = imgviz.label2rgb(
        mask_array,
        img_gray_array,
        font_size=15,
        label_names=['_background_','road'],
        loc="rb",
    )
    imgviz.io.imsave(os.path.join(save_dir,file_name+'_fusion'+'.'+os.path.basename(original_path).split('.')[-1]), viz)

def json2mask(json_path,save_mask_path):
    label_file = labelme.LabelFile(filename=json_path)
    img = labelme.utils.img_data_to_arr(label_file.imageData)
    lbl, _ = labelme.utils.shapes_to_label(
        img_shape=img.shape,
        shapes=label_file.shapes,
        label_name_to_value={'_background_':0,'road':1},
    )
    labelme.utils.lblsave(save_mask_path, lbl)

def mask2graymask(input_folder,output_folder):
    """
    首先遍历文件夹中的所有RGB图像文件，并将它们转换为灰度图像。
    然后，我们统计所有灰度图像的像素值种类和个数，并生成映射规则。
    最后，我们将映射规则应用于每张灰度图像，确保相同的像素值种类被映射为相同的值。
    需要修改：
    input_folder 和 output_folder 路径
    """
    # input_folder = r"F:\datasets\Team_segmentation_datasets\motion_static\mask_png"
    # output_folder = r"F:\datasets\Team_segmentation_datasets\motion_static\gray"

    # 确保输出文件夹存在，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    # 列出文件夹中的RGB图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    # 存储所有灰度图像的像素值
    all_pixel_values = []

    # 遍历图像文件
    for index, image_file in enumerate(image_files):
        # 输出当前正在处理的图片文件名
        print(f"Processing image {index + 1}/{len(image_files)}: {image_file}")

        def cv2_imread(file_path):
            img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            return img
        # 读取RGB图像并转换为灰度图像
        rgb_image = cv2.imread(os.path.join(input_folder, image_file), cv2.IMREAD_COLOR)
        # rgb_image = cv2_imread(os.path.join(input_folder, image_file))
        # print(os.path.join(input_folder, image_file.encode('utf-8').decode('utf-8')),'\n',rgb_image)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # 统计像素值种类和数量
        all_pixel_values.extend(np.unique(gray_image))

        # 保存灰度图像
        cv2.imwrite(os.path.join(output_folder, image_file.encode('utf-8').decode('utf-8')), gray_image)
        # cv2.imencode(ext='.jpg', img=gray_image)[1].tofile(os.path.join(output_folder, image_file))
    #
    # 统计所有灰度图像的像素值种类和个数
    all_pixel_values = sorted(list(set(all_pixel_values)))
    num_classes = len(all_pixel_values)

    # 生成映射规则
    mapping = {pixel_value: i for i, pixel_value in enumerate(all_pixel_values)}

    print("映射规则:")
    for pixel_value, mapped_value in mapping.items():
        print(f"Original: {pixel_value}, Mapped: {mapped_value}")

    # 将映射规则应用于每张灰度图像
    for index, image_file in enumerate(image_files):
        # 输出当前正在处理的图片文件名
        print(f"Mapping image {index + 1}/{len(image_files)}: {image_file}")

        # 读取灰度图像并映射像素值
        gray_image = cv2.imread(os.path.join(output_folder, image_file), cv2.IMREAD_GRAYSCALE)
        mapped_image = np.vectorize(mapping.get)(gray_image)
        # 打印原始值和映射后的新值
        for unique_value in np.unique(gray_image):
            mapped_value = mapping[unique_value]
            print(f"Original: {unique_value}, Mapped: {mapped_value}")
        # 保存映射后的图像
        cv2.imwrite(os.path.join(output_folder, image_file), mapped_image)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default="D:/Dataset/road_dataset/annotations", help="input annotated directory")
    parser.add_argument("--output_dir", default="D:/Dataset/road_dataset", help="output dataset directory")
    parser.add_argument("--labels", default="D:/Dataset/road_dataset/label.txt", help="labels file")
    args = parser.parse_args()
    args.noviz = False

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    os.makedirs(osp.join(args.output_dir, "SegmentationClassPNG"))

    if not args.noviz:
        os.makedirs(
            osp.join(args.output_dir, "SegmentationClassVisualization")
        )
    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_lbl_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".npy"
        )
        out_png_file = osp.join(
            args.output_dir, "SegmentationClassPNG", base + ".png"
        )
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )

        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        labelme.utils.lblsave(out_png_file, lbl)

        np.save(out_lbl_file, lbl)

        if not args.noviz:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(out_viz_file, viz)


if __name__ == "__main__":
    # main()
    # visualization('标注数据示例.jpg','graymask/a_mask.png','')
    # json2mask('标注数据示例.json','标注数据示例_mask.png')
    # json2mask('result.json', 'result_mask.png')
    mask2graymask('mask','graymask')