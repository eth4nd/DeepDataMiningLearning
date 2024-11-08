import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
#from DeepDataMiningLearning.detection.coco_utils import get_coco
# from DeepDataMiningLearning.detection import trainutils
from DeepDataMiningLearning.detection.transforms import PILToTensor, ToDtype, Compose, Resize
import os
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import csv
from pathlib import Path

WrapNewDict = False

def get_transformsimple(is_train: bool):
    """
    Returns a composed set of transformations for the KITTI dataset.
    
    Args:
        is_train (bool): Whether to apply training transformations (e.g., augmentation).
    
    Returns:
        Callable: A composition of transformations to apply to the dataset.
    """
    from DeepDataMiningLearning.detection.transforms import Compose, Resize, PILToTensor, ToDtype

    transforms = [
        Resize((370, 1224)),  # Ensure consistent dimensions
        PILToTensor(),        # Convert image to a tensor
        ToDtype(torch.float, scale=True),  # Convert dtype to float
    ]
    
    if is_train:
        # Add any training-specific transformations here, e.g., data augmentation
        pass  # Example: transforms.append(RandomHorizontalFlip(p=0.5))
    
    return Compose(transforms)

class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str,
                 train: bool = True,
                 split: str = 'train',  # 'val' or 'test'
                 transform: Optional[Callable] = None,
                 image_dir: str = "image_2", 
                 labels_dir: str = "label_2"):
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self.transform = transform
        self._location = "training" if self.train else "testing"
        self.image_dir_name = image_dir
        self.labels_dir_name = labels_dir

        # Determine the split file path
        split_dir = Path(self.root) / 'ImageSets' / f"{split}.txt"

        # Check if the split file exists
        if split_dir.exists():
            self.sample_id_list = [x.strip() for x in open(split_dir).readlines()]
        else:
            raise FileNotFoundError(f"[ERROR] Split file not found: {split_dir}. Ensure the file exists and is correctly named.")

        self.root_split_path = os.path.join(self.root, "raw", self._location)

        # Validate that all necessary image and label files exist
        self._validate_files()

        self.INSTANCE_CATEGORY_NAMES = [
            '__background__', 'Car', 'Van', 'Truck', 'Pedestrian', 
            'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare'
        ]
        self.INSTANCE2id = {
            '__background__': 0, 'Car': 1, 'Van': 2, 'Truck': 3, 
            'Pedestrian': 4, 'Person_sitting': 5, 'Cyclist': 6, 
            'Tram': 7, 'Misc': 8, 'DontCare': 9
        }
        self.numclass = 9  # Including background, excluding 'DontCare'

    def _validate_files(self):
        """
        Validate that all files referenced in self.sample_id_list exist in the expected locations.
        Logs missing or misaligned files and raises an error if critical files are missing.
        """
        missing_images = []
        missing_labels = []

        for sample_id in self.sample_id_list:
            image_file = Path(self.root_split_path) / self.image_dir_name / f"{sample_id}.png"
            label_file = Path(self.root_split_path) / self.labels_dir_name / f"{sample_id}.txt"

            if not image_file.exists():
                missing_images.append(str(image_file))
            if self.train and not label_file.exists():
                missing_labels.append(str(label_file))

        # Log missing files
        if missing_images:
            print(f"[WARNING] Missing image files: {len(missing_images)}")
            for img in missing_images[:5]:  # Show a preview of missing files
                print(f"  - {img}")
        if missing_labels:
            print(f"[WARNING] Missing label files: {len(missing_labels)}")
            for lbl in missing_labels[:5]:  # Show a preview of missing files
                print(f"  - {lbl}")

        # Raise an error if critical files are missing
        if missing_images:
            raise FileNotFoundError(f"{len(missing_images)} image files are missing. Check dataset paths.")
        if self.train and missing_labels:
            raise FileNotFoundError(f"{len(missing_labels)} label files are missing. Check dataset paths.")

    def get_image(self, idx):
        img_file = Path(self.root_split_path) / self.image_dir_name / ('%s.png' % idx)
        print(f"Checking image path: {img_file}")
        assert img_file.exists()
        image = Image.open(img_file)
        return image
    
    def get_label(self, idx):
        label_file = Path(self.root_split_path) / self.labels_dir_name / ('%s.txt' % idx)
        assert label_file.exists(), f"[ERROR] Label file {label_file} does not exist!"
        target = []
        with open(label_file) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "image_id": idx,
                        "type": line[0],
                        "truncated": float(line[1]),
                        "occluded": int(line[2]),
                        "alpha": float(line[3]),
                        "bbox": [float(x) for x in line[4:8]],
                        "dimensions": [float(x) for x in line[8:11]],
                        "location": [float(x) for x in line[11:14]],
                        "rotation_y": float(line[14]),
                    }
                )
        return target

    def convert_target(self, image_id, target):
        num_objs = len(target)
        boxes = []
        labels = []
        for i in range(num_objs):
            bbox = target[i]['bbox'] ##represent the pixel locations of the top-left and bottom-right corners of the bounding box
            xmin = bbox[0]
            xmax = bbox[2]
            ymin = bbox[1]
            ymax = bbox[3]
            objecttype=target[i]['type']
            if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:
                labelid = self.INSTANCE2id[objecttype]
                labels.append(labelid)
                boxes.append([xmin, ymin, xmax, ymax]) #required for Torch [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
        num_objs = len(labels) #update num_objs
        newtarget = {}
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = int(image_id)
        #image_id = torch.tensor([image_id])
        #Important!!! do not make image_id a tensor, otherwise the coco evaluation will send error.
        #image_id = torch.tensor(image_id)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        if num_objs >0:
            newtarget["boxes"] = boxes
            newtarget["labels"] = labels
            #newtarget["masks"] = masks
            newtarget["image_id"] = image_id
            newtarget["area"] = area
            newtarget["iscrowd"] = iscrowd
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
            target['labels'] = labels #torch.as_tensor(np.array(labels), dtype=torch.int64)#empty
            target['image_id'] =image_id
            target["area"] = area #torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
            target["iscrowd"] = iscrowd #torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty
        return newtarget

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a dictionary with bounding box and label information.
        """
        # Ensure the index is within bounds
        if index >= len(self.sample_id_list):  # Handle boundary properly
            raise IndexError(f"Index {index} out of range for dataset of length {len(self.sample_id_list)}.")
        
        # Get the image ID and image
        imageidx = self.sample_id_list[index]
        image = self.get_image(imageidx)

        # Debug: Check if the image is loaded correctly
        if image is None:
            raise ValueError(f"[ERROR] Image at index {index} with ID {imageidx} could not be loaded.")

        # Get the target if in training mode
        if self.train:
            target = self.get_label(imageidx)
            if target is None or len(target) == 0:
                raise ValueError(f"[ERROR] No target data found for image ID {imageidx}. Check label file.")

            target = self.convert_target(imageidx, target)
            if target is None or not isinstance(target, dict) or "boxes" not in target:
                raise ValueError(f"[ERROR] Target conversion failed for image ID {imageidx}.")
        else:
            target = None

        # Optionally wrap the target in a dictionary for annotations
        if WrapNewDict:
            target = dict(image_id=imageidx, annotations=target)

        # Apply transformations, if specified
        if self.transform:
            image, target = self.transform(image, target)
        # Return the image and target
        return image, target

    def _parse_target(self, index: int) -> List:
        target = []
        labelfile = self.targets[index]
        full_name = os.path.basename(labelfile)
        file_name = os.path.splitext(full_name)
        imageidx=int(file_name[0]) #filename index 000001
        #img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        #assert img_file.exists()

        with open(labelfile) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "image_id": imageidx, #new added to ref the filename
                        "type": line[0], #one of the following: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', or 'DontCare'. 'DontCare' is used for objects that are present but ignored for evaluation.
                        "truncated": float(line[1]), #A value of 0.0 means the object is fully visible, and 1.0 means the object is completely outside the image frame.
                        "occluded": int(line[2]), #integer value indicating the degree of occlusion, where 0 means fully visible, and higher values indicate increasing levels of occlusion.
                        "alpha": float(line[3]), #The observation angle of the object in radians, relative to the camera. It is the angle between the object's heading direction and the positive x-axis of the camera.
                        "bbox": [float(x) for x in line[4:8]], #represent the pixel locations of the top-left and bottom-right corners of the bounding box
                        "dimensions": [float(x) for x in line[8:11]], #3D dimensions of the object (height, width, and length) in meters
                        "location": [float(x) for x in line[11:14]], #3D location of the object's centroid in the camera coordinate system (in meters)
                        "rotation_y": float(line[14]), #The rotation of the object around the y-axis in camera coordinates, in radians.
                    }
                )
            #Convert to the required format by Torch
            num_objs = len(target)
            boxes = []
            labels = []
            for i in range(num_objs):
                bbox = target[i]['bbox']
                xmin = bbox[0]
                xmax = bbox[2]
                ymin = bbox[1]
                ymax = bbox[3]
                objecttype=target[i]['type']
                #if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:
                labelid = self.INSTANCE2id[objecttype]
                labels.append(labelid)
                boxes.append([xmin, ymin, xmax, ymax]) #required for Torch [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            num_objs = len(labels) #update num_objs
            newtarget = {}
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([index])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            if num_objs >0:
                newtarget["boxes"] = boxes
                newtarget["labels"] = labels
                #newtarget["masks"] = masks
                newtarget["image_id"] = image_id
                newtarget["area"] = area
                newtarget["iscrowd"] = iscrowd
            else:
                #negative example, ref: https://github.com/pytorch/vision/issues/2144
                newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
                target['labels'] = labels #torch.as_tensor(np.array(labels), dtype=torch.int64)#empty
                target['image_id'] =image_id
                target["area"] = area #torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
                target["iscrowd"] = iscrowd #torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty
        return newtarget, imageidx

    def __len__(self) -> int:
        return len(self.sample_id_list)#(self.images)

class MyKittiDetection(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 image_dir: str = "image_2", 
                 labels_dir: str = "label_2"):
        self.images = []
        self.targets = []
        self.root = root
        self.train = train
        self.transform = transform
        self._location = "training" if self.train else "testing"
        self.image_dir_name = image_dir
        self.labels_dir_name = labels_dir
        # load all image files, sorting them to
        # ensure that they are aligned
        image_dir = os.path.join(self.root, "raw", self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self.root, "raw", self._location, self.labels_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))
        #self.imgs = list(sorted(os.listdir(os.path.join(self.root, "PNGImages"))))
        self.INSTANCE_CATEGORY_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
        self.INSTANCE2id = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian':4, 'Person_sitting':5, 'Cyclist':6, 'Tram':7, 'Misc':8, 'DontCare':9} #background is 0
        self.id2INSTANCE = {v: k for k, v in self.INSTANCE2id.items()}
        self.numclass = 9 #including background, excluding the 'DontCare'

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            tuple: (image, target), where
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float

        """
        #img, target = super().__getitem__(idx)
        if index>len(self.images):
            print("Index out-of-range")
            image = None
        else:
            image = Image.open(self.images[index])
            target, image_id = self._parse_target(index) if self.train else None

        if WrapNewDict:
            target = dict(image_id=image_id, annotations=target) #new changes
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def _parse_target(self, index: int) -> List:
        target = []
        labelfile = self.targets[index]
        full_name = os.path.basename(labelfile)
        file_name = os.path.splitext(full_name)
        imageidx=int(file_name[0]) #filename index 000001
        #img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        #assert img_file.exists()

        with open(labelfile) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                target.append(
                    {
                        "image_id": imageidx, #new added to ref the filename
                        "type": line[0], #one of the following: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', or 'DontCare'. 'DontCare' is used for objects that are present but ignored for evaluation.
                        "truncated": float(line[1]), #A value of 0.0 means the object is fully visible, and 1.0 means the object is completely outside the image frame.
                        "occluded": int(line[2]), #integer value indicating the degree of occlusion, where 0 means fully visible, and higher values indicate increasing levels of occlusion.
                        "alpha": float(line[3]), #The observation angle of the object in radians, relative to the camera. It is the angle between the object's heading direction and the positive x-axis of the camera.
                        "bbox": [float(x) for x in line[4:8]], #represent the pixel locations of the top-left and bottom-right corners of the bounding box
                        "dimensions": [float(x) for x in line[8:11]], #3D dimensions of the object (height, width, and length) in meters
                        "location": [float(x) for x in line[11:14]], #3D location of the object's centroid in the camera coordinate system (in meters)
                        "rotation_y": float(line[14]), #The rotation of the object around the y-axis in camera coordinates, in radians.
                    }
                )
            #Convert to the required format by Torch
            num_objs = len(target)
            boxes = []
            labels = []
            for i in range(num_objs):
                bbox = target[i]['bbox']
                xmin = bbox[0]
                xmax = bbox[2]
                ymin = bbox[1]
                ymax = bbox[3]
                objecttype=target[i]['type']
                #if objecttype != 'DontCare' and xmax-xmin>0 and ymax-ymin>0:
                labelid = self.INSTANCE2id[objecttype]
                labels.append(labelid)
                boxes.append([xmin, ymin, xmax, ymax]) #required for Torch [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
            num_objs = len(labels) #update num_objs
            newtarget = {}
            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.as_tensor(labels, dtype=torch.int64)
            image_id = torch.tensor([index])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            if num_objs >0:
                newtarget["boxes"] = boxes
                newtarget["labels"] = labels
                #newtarget["masks"] = masks
                newtarget["image_id"] = image_id
                newtarget["area"] = area
                newtarget["iscrowd"] = iscrowd
            else:
                #negative example, ref: https://github.com/pytorch/vision/issues/2144
                newtarget['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
                target['labels'] = labels #torch.as_tensor(np.array(labels), dtype=torch.int64)#empty
                target['image_id'] =image_id
                target["area"] = area #torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
                target["iscrowd"] = iscrowd #torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty
        return newtarget, imageidx

    def __len__(self) -> int:
        return len(self.images)
 

def countobjects(alltypes):
    counter = {}
    for type in alltypes:
        if type not in counter:
            counter[type] = 0
        counter[type] += 1
    return counter

# import transforms as T
# # Define get_transformsimple at the module level
# def get_transformsimple(train):
#     transforms = []

#     def debug_transform(image, target):
#         # Print the size of the image before and after resizing
#         print(f"Before resizing: {image.size}")
#         image, target = T.Resize((512, 512))(image, target)

#         # Check if the image is a PIL Image or Tensor and print accordingly
#         if isinstance(image, Image.Image):  # Check if it's a PIL Image
#             print(f"After resizing (PIL Image): {image.size}")
#         elif isinstance(image, torch.Tensor):  # Check if it's a PyTorch Tensor
#             print(f"After resizing (Tensor): {image.size()}")

#         return image, target

#     # Add the debug_transform as the first step to check the image size
#     transforms.append(debug_transform)

#     transforms.append(T.PILToTensor())
#     transforms.append(T.ToDtype(torch.float, scale=True))

#     return T.Compose(transforms)

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    
    class args:
        data_augmentation = 'hflip'
        backend = 'PIL'
        use_v2 = False
        weights = ''
        test_only = False
    
    rootPath = '/data/cmpe258-sp24/013978029/dataset/Kitti'
    is_train = True
    kittidataset = MyKittiDetection(rootPath, train=True, transform=get_transformsimple(is_train))
    print(kittidataset.INSTANCE_CATEGORY_NAMES)
    print("Dataset len:", len(kittidataset))
    img, target = kittidataset[0]
    print(img.shape) #CHW format
    print(target.keys()) #['boxes', 'labels', 'image_id', 'area', 'iscrowd']
    print(target['boxes'].shape)  #torch.Size([3, 4]) n,4
    imgdata=img.permute(1, 2, 0) #CHW -> HWC
    #plt.imshow(imgdata)

    # labels in training set
    if WrapNewDict:
        alltargets = [target['annotations'] for _, target in kittidataset]
        print(target['annotations'][0].keys()) #['type', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y']
    else:
        alltargets = [target['labels'] for _, target in kittidataset]
    # typelist=[]
    # for target in alltargets: #each target is [N,4]
    #     for object in target: #list
    #         typelist.append(object['type'])
    # #alltypes = [target['type'] for target in alltargets]
    # counter=countobjects(typelist)
    # print(counter)
    #{'Cyclist': 1627, 'Pedestrian': 4487, 'Car': 28742, 'DontCare': 11295, 'Van': 2914, 'Misc': 973, 'Truck': 1094, 'Tram': 511, 'Person_sitting': 222}
