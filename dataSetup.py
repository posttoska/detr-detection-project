from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch
import matplotlib.pyplot as plt


CLASS2IDX = {'car': 0}
# maps class names (strings) → class indices (integers)
IDX2CLASS = {v: k for k, v in CLASS2IDX.items()}

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098],
          [0.929, 0.694, 0.125], [0.494, 0.184, 0.556],
          [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
COLORS *= 100


# visualize image with bbox'es and labels
def plot_im_with_boxes(im, boxes, probs=None, ax=None):
    # if no axis (ax) is provided grab the current axis
    if ax is None:
        plt.imshow(im)
        ax = plt.gca()

    # unpack the coordinates
    for i, b in enumerate(boxes.tolist()):
        xmin, ymin, xmax, ymax = b

        # draw a colored rectangle for the bounding box
        patch = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, color=COLORS[i], linewidth=2)
        ax.add_patch(patch)

        # ckeck for prediction scores
        if probs is not None:
            # for visualizing true labels or simplified outputs
            if probs.ndim == 1:
                cl = probs[i].item()
                text = f'{IDX2CLASS[cl]}'
            # for model predictions during inference
            else:
                cl = probs[i].argmax().item()
        else:
            text = ''

        # display the class label as text on top of the bounding box
        ax.text(xmin, ymin, text, fontsize=7,
                bbox=dict(facecolor='yellow', alpha=0.5))


# class to load and preprocess a folder of images
class DetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_csv, transform=None):
        # dir for images
        self.image_dir = image_dir
        # annotations
        self.annotations = pd.read_csv(annotation_csv)
        # transforms
        self.transform = transform
        # group by filename for fast lookup
        self.grouped = self.annotations.groupby("filename")
        # unique image list
        self.image_files = list(self.grouped.groups.keys())

    # PyTorch's Dataset interface
    def __len__(self):
        return len(self.image_files)

    # method for extracting image and its name
    def __getitem__(self, idx):
        # retrieve filename of the image
        filename = self.image_files[idx]
        # build the absolute path
        image_path = os.path.join(self.image_dir, os.path.basename(filename))
        # loads the image and ensures it's in RGB format
        image = Image.open(image_path).convert("RGB")
        # we wiil use that for transforms
        if self.transform:
            image = self.transform(image)

        # extract boxes and labels
        records = self.grouped.get_group(filename)
        boxes = records[["xmin", "ymin", "xmax", "ymax"]].values
        labels = records["class"].map(CLASS2IDX).values

        # convert to torch tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return image, {"boxes": boxes, "labels": labels}

def collate_fn(inputs):
    # extract the image tensor
    input_ = torch.stack([i[0] for i in inputs])
    # extract the labels (classes) and bounding boxes from each sample
    classes = tuple(i[1]["labels"] for i in inputs)
    boxes = tuple(i[1]["boxes"] for i in inputs)
    return input_, (classes, boxes)

from torch.utils.data import DataLoader
from torchvision import transforms
import os

# path to images and .csv file
image_dir = r"C:\Users\posttoska\Documents\jnotebook\ViT\data_ViT\train\images"
csv_path  = r"C:\Users\posttoska\Documents\jnotebook\ViT\data_ViT\train\train.csv"

# transform rule
transform = transforms.Compose([
    # use nearest-neighbor upscaling
    transforms.Resize((2048, 2048), interpolation=transforms.InterpolationMode.NEAREST),
    # convert to tensor
    transforms.ToTensor()
])

# initialize the dataset instance
train_dataset = DetectionDataset(image_dir, csv_path, transform=transform)
dataloader    = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)