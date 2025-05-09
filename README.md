# Microscopy computational tools

This repository contains scripts to identify cell centers and obtain embeddings of cells or images with pretrained machine learning models. The scripts can be run locally or on the [Hail Batch service](https://hail.is/docs/batch/service.html).

The scripts are easy to modify, e.g., to add a new model.

## Installation and dependencies

We use [Pixi](https://pixi.sh/latest/#installation)] for dependency management. Run `pixi install` to download the dependencies.

## Cell centers or embbeddings of images

The script `image_embedding.py` runs a model on an entire image or set of images.  Currently, the following models are supported:

* [Cellpose](https://github.com/MouseLand/cellpose)
* [UniDINO](https://github.com/Bayer-Group/uniDINO/)

The output is consolidated into a single csv, tsv or HDF5 file per plate.

```
usage: image_embedding.py [@h]
                          {cellpose,unidino} model_path plate_path channel_names channel_substrings num_workers [num_processes] [process_idx] [output_file]

per cell embedding

positional arguments:
  {cellpose,unidino}
  model_path          model
  plate_path          folder on the local file system containing the images of one plate
  channel_names       comma seperated names of channels
  channel_substrings  comma seperated substrings of filename to identify channels
  num_workers         number of processes for loading data
  num_processes       number of parallel runs of this script (default: 1)
  process_idx         index of parallel run {0, 1, ..., num_processes -1} (default: 0)
  output_file         output filename (default: embedding.tsv)

options:
  @h, @@help          show this help message and exit
```

Model_path is the filename of the model. Cellpose has a hardcoded model path and ignores this parameter. For uniDINO, it should point to the file `uniDINO.pth` that can be obtained from [Zenodo](https://zenodo.org/records/14988837).

The channel names should match the model input. Cellpose expects the DNA channel. UniDINO is channel agnostic and creates a 384\*num_channels dimensional embedding for each channel. Channels are sorted alphabetically to create consistent output, so the first 384 components correspond to the alphabetically first channel.

The script uses a Torch dataloader. If num_workers > 0, it launches background processes for reading images.

Running multiple instances in parallel on the same machine can help saturate the GPU. Below is an example that uses the convenient utility [moreutils](https://joeyh.name/code/moreutils/) parallel. When num_processes > 1, the output filename is automatically modified to contain the process index. For example, with three parallel processes, the output files will be named embedding_0_3.tsv, embedding_1_3.tsv, and embedding_2_3.tsv. These files can be concatenated with `cat embedding_* > embedding.tsv`.

Example commands:
```
pixi run python image_embedding.py cellpose - BR00123 DNA ch5 0 4
parallel -j 4 pixi run python image_embedding.py cellpose - BR00123 DNA ch5 0 4 -- 0 1 2 3
pixi run python image_embedding.py unidino uniDINO.pth BR00123 DNA,RNA,AGP,ER,Mito -ch5,-ch3,-ch1,-ch4,-ch2 0
```

The output is a tsv file with one line per file containing the filename and embedding. With cellpose, the embedding is split into two columns:

```
file	i	j
r01c01f01p01-ch2sk1fk1fl1.jxl	[156, 468]	[8, 11]
r01c01f09p01-ch2sk1fk1fl1.jxl	[291, 953, 527, 175]	[11, 21, 42, 46]
```

The output file can be read with Pandas via:

```
import pandas as pd
from ast import literal_eval
df = pd.read_csv('embedding.tsv', sep='\t', converters={'i':literal_eval, 'j':literal_eval})
```


## Embbeddings of cells

The script `cell_embedding.py` runs a model on individual cells. Currently, the following models are supported:

* [CellPainting-CNN](https://www.nature.com/articles/s41467-024-45999-1)
* [Dino4Cells](https://www.biorxiv.org/content/10.1101/2023.06.16.545359v1)

The output is consolidated into a single csv, tsv or HDF5 file per plate.

```
usage: cell_embedding.py [@h]
                         {cpcnn,dino4cells} model_path plate_path channel_names channel_substrings centers_path num_workers [output_file]
                         [inspection_file]

per image embedding

positional arguments:
  {cpcnn,dino4cells}
  model_path          model
  plate_path          folder on the local file system containing the images of one plate
  channel_names       comma seperated names of channels
  channel_substrings  comma seperated substrings of filename to identify channels
  centers_path        filename with cell centers
  num_workers         number of processes for loading data
  output_file         output filename (default: embedding.tsv)
  inspection_file     output filename with image crops for manual inspection (default: None)

options:
  @h, @@help          show this help message and exit
```

The model path should point to [DINO_cell_painting_base_checkpoint.pth](https://zenodo.org/records/8061428) (dino4cells) or [Cell_Painting_CNN_v1.hdf5](https://zenodo.org/records/7114558) (CP-CNN).

The centers path is the csv or tsv output of cellpose.

The script uses a Torch dataloader. If num_workers > 0, it launches background processes for reading images and creating cell crops.


Example command:
```
python cell_embedding.py cpcnn Cell_Painting_CNN_v1.hdf5 BR00123 DNA,RNA,AGP,ER,Mito -ch5,-ch3,-ch1,-ch4,-ch2 cellpose_output.csv 0
```

Depending on the output filename, the output file will be csv, tsv or HDF5. For csv or tsv output, there will be one line per cell containing the filename, i,j coordinates of cell centers, and the embedding:

```
file    i    j    embedding
r01c01f01p01-ch5sk1fk1fl1.jxl	2	51	['1.236e+00', '2.663e+00', .... ]
r01c01f01p01-ch5sk1fk1fl1.jxl	8	154	['1.539e+00', '2.860e+00', .... ]
```

For HDF5 output, under meta\filename is a list of filenames. Under cpcnn or dino4cells is a matrix where each cell is one row, the first column corresponds to the filename index, the second and third column are the i/j coordinates, and the remaining columns are the embedding.
