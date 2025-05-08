import sys
import glob
import argparse
import numpy as np
import pandas as pd
import pillow_jxl
from PIL import Image
from cellpose import models
from cellpose.dynamics import get_centers
from scipy.ndimage import find_objects

SCALE = 2**16 - 1

def rescale(arr):
    arr = arr.copy()
    arr = arr.astype(np.double)
    arr -= np.min(arr)
    arr = (arr * SCALE / np.max(arr)).astype(np.uint16)
    return arr

def log_scale(arr):
    return rescale( np.log1p(arr) )
    return arr

def cell_outlines(images_folder, files):
    output = []
    model = models.Cellpose(gpu=True, model_type='nuclei')
    for file in files:
        try:
            im = Image.open(f'{images_folder}/{file}')
            im_resized = np.array(im.resize((512,512), Image.Resampling.NEAREST))
            if np.max(im_resized) - np.min(im_resized) < 1000:
                continue
            imgs = [log_scale(im_resized)]
            masks, flows, styles, diams = model.eval(imgs, diameter=25, channels=[0,0],
                                                     flow_threshold=0.4, do_3D=False)

            slices = find_objects(masks[0])
            # turn slices into array
            slices = np.array([
                np.array([i, si[0].start, si[0].stop, si[1].start, si[1].stop])
                for i, si in enumerate(slices)
                if si is not None
            ])
            centers, _ = get_centers(masks[0], slices)
            scale_x = im.size[0] / 512
            scale_y = im.size[1] / 512
            for center in centers:
                output.append( {'file': file, 'i': round(scale_x * center[0]), 'j': round(scale_y * center[1])} )
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print('failed', file, e)
    return output


parser = argparse.ArgumentParser(description='run cellpose', prefix_chars='@')
parser.add_argument('plate_path', type=str, help='folder containing images')
parser.add_argument('dna_channel_substring', type=str, help='substring of filename to identify DNA channel')
parser.add_argument('num_processes', type=int, help='number of processes')
parser.add_argument('process_idx', type=int, help='process index')
args = parser.parse_args()


plate_path      = args.plate_path
channel_filter  = args.dna_channel_substring
num_processes   = args.num_processes
process_idx     = args.process_idx
output_file     = f'cellpose_{process_idx}_{num_processes}.csv'
add_header      = num_processes == 1

files = glob.glob(f'{plate_path}*{channel_filter}*')
files = [f.removeprefix(plate_path) for f in files]
files = sorted(files)[process_idx::num_processes]

output = cell_outlines(plate_path, files)

df = pd.DataFrame(output)
df = df.groupby('file').agg(list).reset_index()
df[['file', 'i', 'j']].to_csv(output_file, sep='\t', header=add_header, index=False)