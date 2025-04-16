import hailtop.batch as hb
from shlex import quote
import yaml
import argparse
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description='run embedding model on Hail Batch', prefix_chars='@')
parser.add_argument('model', type=str, choices=['dino4cells', 'cpcnn'])
parser.add_argument('plate_path', type=str, help='folder containing plates')
parser.add_argument('output_folder', type=str, help='output folder')
parser.add_argument('channel_names', type=str, help='comma seperated names of channels')
parser.add_argument('channel_substrings', type=str, help='comma seperated substrings of filename to identify channels')
parser.add_argument('centers_path', type=str, help='path to cell centers')
parser.add_argument('plates', type=str, nargs='+', help='plate names')
args = parser.parse_args()

plate_path = urlparse(args.plate_path)
bucket_name = plate_path.netloc
input_folder = plate_path.path.rstrip('/')
output_folder = args.output_folder.rstrip('/')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

backend = hb.ServiceBackend(billing_project=config['hail-batch']['billing-project'],
                            remote_tmpdir=config['hail-batch']['remote-tmpdir'],
                            regions=config['hail-batch']['regions'])

b = hb.Batch(backend=backend, name=f'embedding {args.model}')
for plate in args.plates:
    j = b.new_job(name=f'embedding {args.model} {plate}')
    j.cloudfuse(bucket_name, '/images')
    j._machine_type = config['embedding']['machine-type']
    j.storage('30Gi') # should be large enough for pixi (12 GB), model and tsv output (not for images)

    model_weights = b.read_input(config['embedding']['model-weights'][args.model])
    centers_file = b.read_input(args.centers_path.format(plate=plate))

    num_workers = config['embedding']['num-workers']
    image_folder = f'{input_folder}/{plate}/'

    j.command('apt update')
    j.command('apt install -y git curl moreutils')
    j.command('git clone https://github.com/atgu/microscopy_computational_tools.git')
    j.command('cd microscopy_computational_tools')
    j.command('curl -fsSL https://pixi.sh/install.sh | sh')
    j.command('export PATH=/root/.pixi/bin:$PATH')
    j.command('pixi install')
    j.command(f'pixi run python embeddings/run_model.py {args.model} {model_weights} /images/{quote(image_folder)} {quote(args.channel_names)} {quote(args.channel_substrings)} {quote(centers_file)} {num_workers} embedding.tsv crops.png')
    j.command(f'mv embedding.tsv {j.ofile1}')
    j.command(f'mv crops.png {j.ofile2}')
    b.write_output(j.ofile1, f'{output_folder}/embedding_{args.model}_{plate}.tsv')
    b.write_output(j.ofile2, f'{output_folder}/embedding_{args.model}_{plate}.png')
b.run()
