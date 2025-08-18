import glob
import os.path
import numpy as np
import pandas as pd
from utils.subimage_inspector import Subimage_inspector

from ast import literal_eval
import argparse
from torch.utils.data import DataLoader
from utils.image_loader import Cell_Data_Set, Cell_Batch_Sampler, Scaling

def cell_embeddings(model_name, model_path, images_folder, centers, output_file, inspection_file, channel_names, channel_substrings, num_workers, averages):
    if model_name == 'dino4cells':
        from models.dino4cells import dino_model
        model = dino_model(model_path)
        num_output_features = 768
        input_channels = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']
        scaling = Scaling.CHANNEL_Z_SCORE
    if model_name == 'dino4cells_small':
        from models.dino4cells import dino_model
        model = dino_model(model_path)
        num_output_features = 384
        input_channels = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']
        scaling = Scaling.CHANNEL_Z_SCORE
    elif model_name == 'cpcnn':
        from models.cpcnn import cpcnn_model
        model = cpcnn_model(args.model_path)
        num_output_features = 672
        input_channels = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']
        scaling = Scaling.CELL_ZERO_ONE

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
        num_incomplete = len(centers) - len(image_groups)
        print(f"WARNING: {num_incomplete} out of {len(centers)} image sets have missing images and will not be processed.")

    ds = Cell_Data_Set(image_groups, centers, scaling)
    bs = Cell_Batch_Sampler(image_groups, centers)
    dataloader = DataLoader(ds, batch_sampler=bs, num_workers=num_workers, pin_memory=True)

    # set up hdf5, tsv, csv, and png output
    if output_file.endswith('.h5') or output_file.endswith('.hdf5'):
        from utils.hdf5writer import embedding_writer
        num_rows = len(ds)
        # the +2 accounts for i/j coordinates
        writer = embedding_writer(output_file, model_name, num_rows, num_output_features + 2, 'f4')
    else:
        from utils.csvwriter import CSVWriter
        writer = CSVWriter(output_file)
        writer.write_header("file i j embedding".split())
    
    if averages: # Create a new writer for averages
        from utils.csvwriter import CSVWriter
        writer_avg = CSVWriter(f'{output_file.rpartition('.')[0]}_avg.tsv')
        csv_header = ['file'] + [f'feature_{idx}' for idx in range(num_output_features)]
        writer_avg.write_header(csv_header)

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
        if averages:
            writer_avg.writerow(dna_imname, embeddings.mean(axis=0))
        
        #if subimage_inspector.current_row > 0:
        #    break
    
    if inspection_file is not None:
        subimage_inspector.save(inspection_file)

    writer.close()
    if averages:
        writer_avg.close()


parser = argparse.ArgumentParser(description='per image embedding', prefix_chars='@')
parser.add_argument('model', type=str, choices=['cpcnn', 'dino4cells', 'dino4cells_small'])
parser.add_argument('model_path', type=str, help='model')
parser.add_argument('plate_path', type=str, help='folder containing images')
parser.add_argument('channel_names', type=str, help='comma seperated names of channels')
parser.add_argument('channel_substrings', type=str, help='comma seperated substrings of filename to identify channels')
parser.add_argument('centers_path', type=str, help='filename with cell centers')
parser.add_argument('num_workers', type=int, help='number of processes for loading data', nargs='?', default=1)
parser.add_argument('output_file', type=str, help='output filename', nargs='?', default='embedding.tsv')
parser.add_argument('inspection_file', type=str, help='output filename with image crops for manual inspection', nargs='?')
parser.add_argument('averages', type=lambda x: x.lower() in ['true'], nargs='?', default=False, help='whether to compute averages (True/False, default=False)')

args = parser.parse_args()

images_folder = args.plate_path
if not images_folder.endswith('/'):
    images_folder = images_folder + '/'


if args.channel_names.count(',') != args.channel_substrings.count(','):
    raise Exception('ERROR: Channel names and substrings should have the same length.')

channel_names      = [s.strip() for s in args.channel_names.split(',')]
channel_substrings = [s.strip() for s in args.channel_substrings.split(',')]

centers = pd.read_table(args.centers_path, index_col='file', sep=None, engine='python')
centers['i'] = centers['i'].apply(literal_eval)
centers['j'] = centers['j'].apply(literal_eval)

cell_embeddings(args.model, args.model_path, images_folder, centers, args.output_file, args.inspection_file, channel_names, channel_substrings, args.num_workers, args.averages)