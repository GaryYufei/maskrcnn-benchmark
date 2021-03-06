from maskrcnn_benchmark.structures.bounding_box import BoxList
import json, os, torch
from PIL import Image

class VGDataset(object):
    def __init__(self, img_dir, vg_ann, class_file, attr_file, transforms=None):
        self.img_dir = img_dir

        self.cls_dict = {"__background__": 0}
        self.cls_list = [None]
        with open(class_file) as cls_file:
            for line in cls_file:
                line = line.strip()
                if ',' in line:
                    line = line.split(',')[0]
                self.cls_list.append(line)
                self.cls_dict[line] = len(self.cls_dict)

        self.attr_dict = {"*empty*": 0}
        with open(attr_file) as attrs_file:
            for line in attrs_file:
                line = line.strip()
                if ',' in line:
                    line = line.split(',')[0]
                self.attr_dict[line] = len(self.attr_dict)

        with open(vg_ann) as ann:
            self.img_obj_list = json.load(ann)
    
        self.transforms = transforms

    def __len__(self):
        return len(self.img_obj_list)

    def map_class_id_to_class_name(self, class_id):
        return self.cls_list[class_id]

    def __getitem__(self, idx):
        image_info = self.img_obj_list[idx]
        # load the image as a PIL Image
        image_path = os.path.join(self.img_dir, image_info['image_name'])
        image = Image.open(image_path).convert("RGB")

        boxlist = self.get_groundtruth(idx)
        
        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)
        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_groundtruth(self, idx):
        image_info = self.img_obj_list[idx]

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes, labels, difficult, attrs = [], [], [], []
        for object_ in image_info['objects']:
            boxes.append([int(object_['x1']), int(object_['y1']), int(object_['x2']), int(object_['y2'])])
            labels.append(self.cls_dict[object_['label']])
            attrs.append(self.attr_dict[object_['attr'][0]])
            difficult.append(0)
        # and labels
        labels = torch.tensor(labels)
        difficult = torch.tensor(difficult)
        attrs = torch.tensor(attrs)

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, (image_info['width'], image_info['height']), mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        boxlist.add_field("difficult", difficult)
        boxlist.add_field("attrs", attrs)

        return boxlist


    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image_info = self.img_obj_list[idx]
        return {"height": image_info['height'], "width": image_info['width']}
