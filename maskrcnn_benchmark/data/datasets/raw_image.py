import json

class RawImageDataset(object):

	def __init__(self, img_dir, image_info_path, transforms=None):
		with open(image_info_path) as ann:
            self.img_obj_list = json.load(ann)['images']
    
        self.transforms = transforms

    def __len__(self):
        return len(self.img_obj_list)

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        image_info = self.img_obj_list[idx]
        return {"height": image_info['height'], "width": image_info['width']}

    def __getitem__(self, idx):
        image_info = self.img_obj_list[idx]
        # load the image as a PIL Image
        image_path = os.path.join(self.img_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        boxlist = BoxList([], (image_info['width'], image_info['height']), mode="xyxy")

        return image, boxlist, image_info['id']
