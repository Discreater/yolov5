import argparse
import pathlib
import random
from typing import List
import xml.etree.ElementTree as ET
import os

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--random-seed', default=False)
    args = parser.parse_args()

    if not args.random_seed:
        random.seed(233)

    dataset_path = args.dataset_path
    dataset_path = pathlib.Path(dataset_path)
    assert dataset_path.is_dir()

    groups = [group for group in dataset_path.iterdir() if group.is_dir()]
    labels = [group.name for group in groups]

    item_list = []

    for group in tqdm(groups, desc="reading xml"):
        xmls: List[pathlib.Path] = [item for item in group.iterdir() if item.suffix == ".xml"]
        for xml in xmls:
            item_list.append(XMLStruct(xml))

    print("shuffling")
    random.shuffle(item_list)
    mid_len = int(len(item_list) * 0.8)
    train_list = item_list[:mid_len]
    valid_list = item_list[mid_len:]

    train_image_path = (dataset_path / "images/train")
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_label_path = (dataset_path / "labels/train")
    train_label_path.mkdir(parents=True, exist_ok=True)
    valid_image_path = (dataset_path / "images/val")
    valid_image_path.mkdir(parents=True, exist_ok=True)
    valid_label_path = (dataset_path / "labels/val")
    valid_label_path.mkdir(parents=True, exist_ok=True)

    item: XMLStruct
    for item in tqdm(train_list, desc="converting train set"):
        os.symlink(item.file_path, train_image_path / item.target_img_path)
        txt_path = train_label_path / item.target_txt_path
        with open(txt_path, "w") as f:
            f.write(item.to_line())
    for item in tqdm(valid_list, desc="converting valid set"):
        os.symlink(item.file_path, valid_image_path / item.target_img_path)
        txt_path = valid_label_path / item.target_txt_path
        with open(txt_path, "w") as f:
            f.write(item.to_line())


class XMLStruct:
    file_path: pathlib.Path
    group_name: str
    file_name: str
    target_txt_path: str
    target_img_path: str
    size: (int, int)
    name: int

    def __init__(self, xml_path: pathlib.Path):
        parent_path = xml_path.parent

        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.group_name = parent_path.name
        self.file_name = root.find("filename").text
        self.file_path = parent_path / (self.file_name + ".jpg")
        target_name = "{}_{}".format(self.group_name, self.file_name)
        self.target_img_path = target_name + ".jpg"
        self.target_txt_path = target_name + ".txt"

        size_node = root.find("size")
        width = int(size_node.find("width").text)
        height = int(size_node.find("height").text)
        self.size = (width, height)

        object_node = root.find("object")
        self.name = object_node.find("name").text

        bndbox_node = object_node.find("bndbox")
        xmax = int(bndbox_node.find("xmax").text)
        xmin = int(bndbox_node.find("xmin").text)
        ymax = int(bndbox_node.find("ymax").text)
        ymin = int(bndbox_node.find("ymin").text)
        self.x_center = (xmax + xmin) / 2 / width
        self.y_center = (ymax + ymin) / 2 / height
        self.b_width = (xmax - xmin) / width
        self.b_height = (ymax - ymin) / height

    def to_line(self):
        return "0 {} {} {} {}\n".format(self.x_center, self.y_center, self.b_width, self.b_height)


if __name__ == '__main__':
    main()
