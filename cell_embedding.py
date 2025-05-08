import glob
import os.path
import numpy as np
import pandas as pd
from utils.subimage_inspector import Subimage_inspector

from ast import literal_eval
import argparse
from torch.utils.data import DataLoader
from utils.image_loader import Cell_Data_Set, Cell_Batch_Sampler

def cell_embeddings(model_name, model_path, images_folder, centers, output_file, inspection_file, channel_names, channel_substrings, num_workers):
    output = []
    if model_name == 'dino4cells':
        from models.dino4cells import dino_model
        model = dino_model(args.model_path)
        num_output_features = 768
        input_channels = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']
    elif model_name == 'cpcnn':
        from models.cpcnn import cpcnn_model
        model = cpcnn_model(args.model_path)
        num_output_features = 672
        input_channels = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']

    missing_channels = set(input_channels) - set(channel_names)
    if len(missing_channels) > 0:
        raise Exception('ERROR: Some channel names are missing. This model requires ' + ','.join(input_channels))
    if 'DNA' not in input_channels:
        raise Exception('ERROR: The DNA channel should be specified')

    # generate image names
    dna_images = [images_folder + imname for imname in centers.index]
    channel_filters = dict(zip(channel_names, channel_substrings))
    other_images = [[dnaim.replace(channel_filters['DNA'], channel_filters[channel]) for dnaim in dna_images]
                     for channel in input_channels if channel != 'DNA']
    image_groups = list(zip(dna_images, *other_images))

    # check that image paths point to actual files
    actual_files = set(glob.glob(images_folder + '*'))
    image_groups = [imgrp for imgrp in image_groups if all(filepath in actual_files for filepath in imgrp)]

    if len(image_groups) != len(centers):
        print(f"WARNING: Found complete image sets for {len(image_groups)} out of {len(centers)}")

    ds = Cell_Data_Set(image_groups, centers)
    bs = Cell_Batch_Sampler(image_groups, centers)
    dataloader = DataLoader(ds, batch_sampler=bs, num_workers=num_workers, pin_memory=True)

    # set up hdf5, tsv, csv, and png output
    if output_file.endswith('.h5') or output_file.endswith('.hdf5'):
        from utils.hdf5writer import embedding_writer
        num_rows = len(ds)
        writer = embedding_writer(output_file, model_name, num_rows, num_output_features, 'f4')
    else:
        from utils.csvwriter import CSVWriter
        writer = CSVWriter(output_file)
        writer.write_header("file i j embedding".split())

    if inspection_file is not None:
        num_sample_crops = 25
        num_channels = len(input_channels)
        num_cells = len(ds)
        resolution = 128
        subimage_inspector = Subimage_inspector(num_cells, num_sample_crops, num_channels, resolution)

    # generate embeddings
    for dna_imnames, centers_i, centers_j, subimages in dataloader:
        dna_imname = dna_imnames[0]
        centers_i = centers_i.tolist()
        centers_j = centers_j.tolist()
        print(dna_imname, len(centers_i))
        if inspection_file is not None:
            subimage_inspector.add(dna_imname, centers_i, centers_j, subimages)
        embeddings = model(subimages)
        writer.writerows(dna_imname, [centers_i, centers_j, embeddings])
        
        #if subimage_inspector.current_row > 0:
        #    break
    
    if inspection_file is not None:
        subimage_inspector.save(inspection_file)

    writer.close()


parser = argparse.ArgumentParser(description='per image embedding', prefix_chars='@')
parser.add_argument('model', type=str, choices=['cpcnn', 'dino4cells'])
parser.add_argument('model_path', type=str, help='model')
parser.add_argument('plate_path', type=str, help='folder containing images')
parser.add_argument('channel_names', type=str, help='comma seperated names of channels')
parser.add_argument('channel_substrings', type=str, help='comma seperated substrings of filename to identify channels')
parser.add_argument('centers_path', type=str, help='filename with cell centers')
parser.add_argument('num_workers', type=int, help='number of processes for loading data')
parser.add_argument('output_file', type=str, help='output filename', nargs='?', default='embedding.tsv')
parser.add_argument('inspection_file', type=str, help='output filename with image crops for manual inspection', nargs='?')
args = parser.parse_args()

images_folder = args.plate_path
if not images_folder.endswith('/'):
    images_folder = images_folder + '/'


if args.channel_names.count(',') != args.channel_substrings.count(','):
    raise Exception('ERROR: Channel names and substrings should have the same length.')

channel_names      = [s.strip() for s in args.channel_names.split(',')]
channel_substrings = [s.strip() for s in args.channel_substrings.split(',')]

centers = pd.read_table(args.centers_path, converters={'i':literal_eval, 'j':literal_eval}, index_col='file')

cell_embeddings(args.model, args.model_path, images_folder, centers, args.output_file, args.inspection_file, channel_names, channel_substrings, args.num_workers)