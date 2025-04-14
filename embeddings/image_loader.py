import os
import numpy as np
import pillow_jxl
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

SCALE = 2**16 - 1

def rescale(arr, axes=[-1, -2]):
    arr = arr.copy()
    arr = arr.astype(np.float32)
    arr -= np.min(arr)
    arr = (arr * SCALE / (np.max(arr) + 1e-8))
    return arr


class Data_Set(Dataset):
    def __init__(self, image_groups, centers, subwindow=128):
        self.image_groups = image_groups
        self.centers = centers
        self.cells_per_image = centers.i.str.len().to_dict()
        self.subwindow = subwindow
        self.file_loaded = None
        self.images = None
        self.cell_idx_to_image_idx = []
        self.cell_idx_to_offset = []
        for image_idx, imgrp in enumerate(image_groups):
            key = os.path.basename(imgrp[0])
            if key not in self.cells_per_image:
                # an image without any detected cells
                continue
            for offset in range(self.cells_per_image[key]):
                self.cell_idx_to_image_idx.append(image_idx)
                self.cell_idx_to_offset.append(offset)
    def __len__(self): 
        num_cells = len(self.cell_idx_to_image_idx)
        return num_cells
    def extract_subimage(self, images, center_i, center_j, subwindow=128):
        output = np.zeros((len(self.images), subwindow, subwindow), dtype=np.float32)  # channels, width, height
        for ichan, im in enumerate(self.images):
            output[ichan, ...] = im[center_i:center_i + subwindow, center_j:center_j + subwindow]
        # rescale each subimage
        output -= np.nanmin(output, axis=(1, 2), keepdims=True)
        out_max = np.nanmax(output, axis=(1, 2), keepdims=True)
        out_max[out_max == 0] = 1
        output /= out_max
        output[ np.isnan(output) ] = 0 # padded values
        return output
    def __getitem__(self, idx):
        image_idx = self.cell_idx_to_image_idx[idx]
        if self.file_loaded != image_idx:
            self.file_loaded = image_idx
            self.images = []
            for filename in self.image_groups[image_idx]:
                try:
                    im = np.asarray(Image.open(filename))
                    self.images.append( im.astype(np.float32) )
                except Exception as e:
                    print(f'WARNING: Loading file failed for {filename}', e)
                    i_max = self.centers['i'].iloc[image_idx].max()
                    j_max = self.centers['j'].iloc[image_idx].max()
                    self.images.append( np.zeros((i_max, j_max), dtype=np.float32) )
            # pad images to simplify subimage extraction
            padwidth = self.subwindow // 2
            # this makes i, j the upper left corner of each subwindow, and prevents out-of-bounds
            self.images = [np.pad(im, (padwidth, self.subwindow - padwidth), constant_values=np.nan) for im in self.images]
        center_i = self.centers['i'].iloc[image_idx][ self.cell_idx_to_offset[idx] ]
        center_j = self.centers['j'].iloc[image_idx][ self.cell_idx_to_offset[idx] ]
        cell = self.extract_subimage(self.images, center_i, center_j, subwindow=128)
        return self.centers.index[image_idx], center_i, center_j, cell


class Batch_Sampler(Sampler):
    def __init__(self, image_groups, centers):
        self.image_groups = image_groups
        self.cells_per_image = centers.i.str.len().to_dict()
    def __iter__(self):
        offset = 0
        for imgrp in self.image_groups:
            key = os.path.basename(imgrp[0])
            if key not in self.cells_per_image:
                # an image without any detected cells
                continue
            batch_size = self.cells_per_image[key]
            yield [offset+i for i in range(batch_size)]
            offset += batch_size