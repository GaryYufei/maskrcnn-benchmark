from maskrcnn_benchmark.structures.bounding_box import BoxList
import json, os, torch
from PIL import Image

class VGDataset(object):
    def __init__(self, img_dir, vg_ann, class_file, transforms=None):
        self.img_dir = img_dir

        self.cls_dict = {}
        with open(class_file) as cls_file:
            for line in cls_file:
                line = line.strip()
                self.cls_dict[line] = len(self.cls_dict)

        with open(vg_ann) as ann:
            self.img_obj_list = json.load(ann)
    
        self.transforms = transforms

    def __len__(self):
        return len(self.img_obj_list)

    def __getitem__(self, idx):
        image_info = self.img_obj_list[idx]
        # load the image as a PIL Image
        image_path = os.path.join(self.img_dir, image_info['image_name'])
        image = Image.open(image_path).convert("RGB")
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes, labels = [], []
        for object_ in image_info['objects']:
            boxes.append([int(object_['x']), int(object_['y']), int(object_['x']) + int(object_['w']) - 1,int(object_['y']) + int(object_['h']) - 1])
            labels.append(self.cls_dict[object_['label']])
        # and labels
        labels = torch.tensor(labels)

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)
        if self.transforms is not None:
            image, boxlist = self.transforms(image, boxlist)
        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image_info = self.img_obj_list[idx]
        return {"height": image_info['height'], "width": image_info['width']}
