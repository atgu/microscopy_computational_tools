import os
import random
import numpy as np
from PIL import Image

class Subimage_inspector:
    """
    Stores sample_size images out of a total of num_cells cells in a file for manual inspection.
    """
    def __init__(self, num_cells, sample_size, num_channels, resolution=128):
        self.rows = sample_size
        self.cols = num_channels
        self.res = resolution
        self.offset = 0
        self.current_row = 0
        self.im = np.zeros((self.rows * self.res, self.cols * self.res), dtype=np.uint16)
        random.seed(0)
        self.sample = set(random.sample(range(num_cells), sample_size))
    def add(self, description, i, j, subimages):
        num_subimages, _, _, _ = subimages.shape
        if num_subimages == 0:
            return
        for i in range(num_subimages):
            if self.offset + i in self.sample:
                subimage = subimages[i, :, :, :] * (2**16 - 1)
                subimage = subimage.cpu().numpy().astype(np.uint16)
                for col in range(self.cols):
                    self.im[self.current_row*self.res:(self.current_row+1)*self.res, col*self.res:(col+1)*self.res] = subimage[col, :, :]
                self.current_row += 1
        self.offset += num_subimages
    def add_overlay(self):
        WHITE = 2**16 - 1
        self.im[::self.res, :] = WHITE
        self.im[:, ::self.res] = WHITE
        for row in range(self.rows):
            for col in range(self.cols):
                for i in range(self.res):
                    self.im[self.res * row + i, self.res * col + i] = WHITE
                    self.im[self.res * row + i, self.res * (col+1) - i - 1] = WHITE
    def save(self, filename):
        self.add_overlay()
        im = Image.fromarray(self.im)
        output_folder = os.path.dirname(filename)
        if output_folder != '':
                os.makedirs(output_folder, exist_ok=True)
        im.save(filename)
