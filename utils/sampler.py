import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torch import Tensor
import csv
import os
from PIL import Image

class OpenSurfaces(Dataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.mean = Tensor([0.48485812, 0.40377784, 0.32280155])
    self.std = Tensor([0.37216536, 0.349832,   0.37452201])
    self.classes = ["Wood", "Concrete", "Granite/marble", "Tile"]
    self.name_to_class = {c:idx for idx, c in enumerate(self.classes)}
    shape_indexes = {idx:[] for idx, c in enumerate(self.classes)}
    file_name = os.path.join(root_dir, 'shapes.csv')

    self.data = []
    with open(file_name,'r') as f:
      csv_reader = csv.reader(f)
      next(csv_reader)
      shape_idx_offset = 0
      for shape_idx, shape in enumerate(csv_reader):
        if shape[2] in self.classes:
          image_path = os.path.join("shapes-cropped", shape[0] + ".png")
          shape_class = self.name_to_class[shape[2]]
          shape_indexes[shape_class].append(shape_idx - shape_idx_offset)
          self.data.append([image_path, shape_class])
    l = len(self.data)
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image_path, shape_class = self.data[idx]
    total_path = os.path.join(self.root_dir, image_path)
    if not os.path.exists(total_path):
      print("No exist: " + image_path)
      return None, None
    image = Image.open(total_path)
    image = image.convert('RGBA')
    if self.transform:
      image = self.transform(image)
      #if is_tensor(image):
      #  image = transforms.Normalize(self.mean, self.std)(image)
    #print(image_path)
    return image, shape_class

class CropSampler(torch.nn.Module):
  def __init__(self, size):
    super().__init__()
    self.size = size

  def forward(self, img):
      width, height = img.size
      if width <  self.size or height <  self.size:
        return -1
      np_image = np.array(img)
      content_mask = (np_image != [255,255,255,0])[:,:,0]
      sample_count_width = int(width/self.size)
      sample_count_height = int(height/self.size)
      #half_size = int(self.size / 2)
      crop_area = self.size*self.size
      crops = []
      for i in range(sample_count_width):
        for j in range(sample_count_height):
          count = np.sum(content_mask[i*self.size : (i + 1)*self.size, j*self.size : (j + 1)*self.size])
          if count == crop_area:
            crops.append((np_image[i * self.size:(i + 1) * self.size, j * self.size:(j + 1) * self.size])[:,:,:3])
      if len(crops) == 0:
        return -1
      return np.array(crops)

sample_dataset = OpenSurfaces(root_dir="../datasets", transform=CropSampler(224))
sample_loader = torch.utils.data.DataLoader(dataset=sample_dataset, batch_size=1)
dataset_path = "../datasets/sample_224"
class_counter = {}
for c in sample_dataset.classes:
  class_counter[c] = 0

for images, label in tqdm(sample_loader):
  if len(images.shape) > 1:
    class_name = sample_dataset.classes[label]
    class_count = class_counter[class_name]
    if class_name == "Granite/marble":
      class_name = "Granite_marble"
    path_folder = os.path.join(dataset_path, class_name)
    for image in images[0]:
       img = Image.fromarray(np.array(image))
       path_image = os.path.join(path_folder, str(class_count) + ".png")
       img.save(path_image,"png")
       class_count += 1
    class_counter[sample_dataset.classes[label]] = class_count
