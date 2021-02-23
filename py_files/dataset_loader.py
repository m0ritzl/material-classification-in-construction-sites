import os 
import time 
import torch 
import csv 
from torch.utils.data import Dataset 
from torch import Tensor, is_tensor 
import torchvision 
import torchvision.transforms as transforms 
from torchvision import transforms, datasets, models 
import numpy as np 
import random 
from tqdm import tqdm 
from torchvision.datasets import CocoDetection
import os
import time
import copy
from PIL import Image
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import Tensor, is_tensor
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

class OpenSurfacesSmall(Dataset):
  def __init__(self, root_dir, n, split=(0.0,1,0), transform=None):
    self.n = n 
    self.split = split
    self.root = root_dir
    self.classes = ["Wood", "Concrete", "Granite_marble", "Tile"]
    self.name_to_class = {c:idx for idx, c in enumerate(self.classes)}
    self.transform = transform

  def __len__(self):
    return int(len(self.classes) * self.n * (self.split[1] - self.split[0]))

  def __getitem__(self, idx):
    n = self.n * (self.split[1] - self.split[0])
    if idx < n:
      c = 0
    elif idx < 2*n:
      c = 1
      idx -= n
    elif idx < 3*n:
      c = 2
      idx -= 2*n
    else:
      c = 3
      idx -= 3*n
    idx = int(self.n * self.split[0] + idx)
    image_path = os.path.join(self.root, self.classes[c], str(idx) + ".png")
    image = Image.open(image_path).convert('RGB')
    if self.transform:
      image = self.transform(image)
    return image, c

class MINC2500(Dataset):
    def __init__(self, root_dir, set_type, split, transform=None):
        self.root_dir = root_dir
        self.set_type = set_type
        self.transform = transform
        # Those values are computed using the script 'get_minc2500_norm.py'
        self.mean = Tensor([0.507207, 0.458292, 0.404162])
        self.std = Tensor([0.254254, 0.252448, 0.266003])

        # Get the material categories from the categories.txt file
        file_name = os.path.join(root_dir, 'categories.txt')
        self.categories = {}
        with open(file_name, 'r') as f:
            for i, line in enumerate(f):
                # The last line char (\n) must be removed
                self.categories[line[:-1]] = i

        # Load the image paths
        self.data = []
        file_name = os.path.join(root_dir, 'labels')
        # For the moment I use only the first split
        file_name = os.path.join(file_name, set_type + str(split) + '.txt')
        with open(file_name, 'r') as f:
            for line in f:
                img_path = line.split(os.sep)
                # The last line char (\n) must be removed
                self.data.append([line[:-1], self.categories[img_path[1]]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data[idx][0])
        image = Image.open(img_path)
        # Sometimes the images are opened as grayscale, so I need to force RGB
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
            if is_tensor(image):
                image = transforms.Normalize(self.mean, self.std)(image)

        return image, self.data[idx][1]

class OpenSurfaces(Dataset):
  def __init__(self, root_dir, split=(0.0, 1.0), transform=None):
    self.root_dir = root_dir
    self.split = split
    self.transform = transform
    self.mean = Tensor([0.48485812, 0.40377784, 0.32280155])
    self.std = Tensor([0.37216536, 0.349832,   0.37452201])
    self.classes = ["Wood", "Metal", "Concrete", 
                    "Brick", "Granite/marble", 
                    "Painted", "Stone", "Tile"]
    self.name_to_class = {c:idx for idx, c in enumerate(self.classes)}
    shape_filter = ['143953','183035','143573','157186','169390','181126','142889','183231','181915','142890','188939','170617','143370','142492','181913','162030','180813','162985','180168',
                    '180979','88823','176183','183232','180998','188346','180140','171005','143652','167741','115717','172468','150521','121120','142958','186081','187209','144669','154576',
                    '166545','139779','169163','159317','143653','171007','92019','142898','178097','89535','160223','143655','170980','94893','140606','180051','175901','140155','109302',
                    '109302','147372','148191','144030','152166','140239','110372','140240','139481','186510','160225','102594','153294','175978','175439','162359','166567','171695','146379',
                    '114542','152262','140603','185901','155259','167573','145483','184548','140083','161131','155274','140034','176012','68542','167568','185143','159725','154152','182853',
                    '157429','120499','150796','164616','182913','189104','176715','156458','111674','117548','161089','79566','111231','150571','154208','153411','186064','78804','143255',
                    '168107','155188','143369','178349','147772','161097','172772','159308','169602','140713','93958','164334','86390','143002','162275','157027','130994','101517','127671',
                    '156666','109246','175933','172234','109689','141722','161093','145481','103181','142833','172662','92403','95157','171687','122987','109559','143823','169549','165597',
                    '123124','172892','107302','177903','188574','114249','127670','171686','150250','136785','144962','109698','42865','116823','65494','173638','109584','101706','42704',
                    '157181','157558','112646','165656','172163','164684','185800','95773','131833','163882','179909','124688','109699','42862','159131','169069','181074','55865','74646',
                    '146687','40921','131546','163957','109003','172642','163292','186587','184260','129470','47228','71644','139338','123284','74652','184457','168968','183945','153065',
                    '104924','146577','155270','91818','175508','157830','47908','109562','171685','172170','162297','47070','86322','175574','45001','166939','50844','169755','50466',
                    '157158','37104','169582','118764','174244','46922','143615','164585','46923','174300','157891','158926','174358','155873','99337','125323','78814','161274','124689',
                    '150043','122606','130994','101517','127671','156666','109246','175933','172234','109689','141722','161093','145481','103181','142833','172662','92403','95157','171687',
                    '122987','109559','143823','169549','165597','123124','172892','107302','177903','188574','114249','127670','171686','150250','136785','144962','109698','42865','116823',
                    '65494','173638','109584','101706','42704','157181','157558','112646','165656','172163','164684','185800','95773','131833','163882','179909','124688','109699','42862','159131',
                    '169069','181074','55865','74646','146687','40921','131546','163957','109003','172642','163292','186587','184260','129470','47228','71644','139338','123284','74652','184457',
                    '168968','183945','153065','104924','146577','155270','91818','175508','157830','47908','109562','171685','172170','162297','47070','86322','175574','45001','166939','50844',
                    '169755','50466','157158','37104','169582','118764','174244','46922','143615','164585','46923','174300','157891','158926','174358','155873','99337','125323','78814','161274',
                    '124689','150043','122606','94249','93485','71643','140960','88130','60069','78955','50464','50455','47817','60064','161670','87042','102296','47067','47819','109602','159745',
                    '101898','91817','153078','91481','172943','154361','71651','175316','118317','112743','51814','71812','85503','85500','85502','175193','158924','116826','62036','152798',
                    '85499','85498','51181','50326','67625','170379','47814','87070','49712','119961','85459','117402','110846','183613','145479','171558','155628','94102','45054','119090',
                    '108448','109603','166739','118248','113511','132258','123873','113512','137645','119218','167073','124574','126549','132498']
    shape_indexes = {idx:[] for idx, c in enumerate(self.classes)}
    file_name = os.path.join(root_dir, 'shapes.csv')
    self.data = []
    with open(file_name,'r') as f:
      csv_reader = csv.reader(f)
      next(csv_reader)
      shape_idx_offset = 0
      for shape_idx, shape in enumerate(csv_reader):
        if shape[0] not in shape_filter:
          image_path = os.path.join("shapes-cropped", shape[0] + ".png")
          shape_class = self.name_to_class[shape[2]]
          shape_indexes[shape_class].append(shape_idx - shape_idx_offset)
          self.data.append([image_path, shape_class])
        else:
          shape_idx_offset += 1
    l = len(self.data)
    if split == (0.0, 1.0):
      self.data = self.data[int(l*split[0]):int(l*split[1])]
    else:
      splitted_data = []
      for shape_class in shape_indexes.keys():
        l = len(shape_indexes[shape_class])
        # Gather all data of the same class
        shapes_with_same_class = [self.data[shape_idx] for shape_idx in shape_indexes[shape_class]]
        # Split the data of a single class and append it to the final dataset
        splitted_data += shapes_with_same_class[int(l*split[0]):int(l*split[1])]
      self.data = splitted_data
    
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
    width, height = image.size
    if width <  32 or height <  32:
      print("Too small: " + image_path)
      return None, None
    if self.transform:
      image = self.transform(image)
      #if is_tensor(image):
      #  image = transforms.Normalize(self.mean, self.std)(image)
    #print(image_path)
    return image, shape_class

class SparseUniformRandomCrop(torch.nn.Module):
  def __init__(self, size):
    super().__init__()
    self.size = size

  def forward(self, img):
    width, height = img.size
    if width <  self.size or height <  self.size:
      raise ValueError("Width of height of image is smaller then crop size")
    np_image = np.array(img)
    content_mask = (np_image != [255,255,255,0])[:,:,0]
    sample_count_width = int(width/self.size)
    sample_count_height = int(height/self.size)
    #half_size = int(self.size / 2)
    crop_area = self.size*self.size
    possible_crops = []
    for i in range(sample_count_width):
      for j in range(sample_count_height):
        count = np.sum(content_mask[i*self.size : (i + 1)*self.size, j*self.size : (j + 1)*self.size])
        if count == crop_area:
          possible_crops.append((i,j))
    if len(possible_crops) == 0:
      return -1
      #raise Exception("Can't find suitable crops.")
    i, j = random.choice(possible_crops)
    crop = (np_image[i * self.size:(i + 1) * self.size, j * self.size:(j + 1) * self.size])[:,:,:3]
    return Image.fromarray(crop)

class SparseUniformFirstCrop(torch.nn.Module):
  def __init__(self, size):
    super().__init__()
    self.size = size

  def forward(self, img):
    width, height = img.size
    if width <  self.size or height <  self.size:
      raise ValueError("Width of height of image is smaller then crop size")
    np_image = np.array(img)
    content_mask = (np_image != [255,255,255,0])[:,:,0]
    sample_count_width = int(width/self.size)
    sample_count_height = int(height/self.size)
    #half_size = int(self.size / 2)
    crop_area = self.size*self.size
    possible_crops = []
    for i in range(sample_count_width):
      for j in range(sample_count_height):
        count = np.sum(content_mask[i*self.size : (i + 1)*self.size, j*self.size : (j + 1)*self.size])
        if count == crop_area:
          possible_crops.append((i,j))
    if len(possible_crops) == 0:
      return -1
      #raise Exception("Can't find suitable crops.")
    i, j = possible_crops[0]
    crop = (np_image[i * self.size:(i + 1) * self.size, j * self.size:(j + 1) * self.size])[:,:,:3]
    return Image.fromarray(crop)
