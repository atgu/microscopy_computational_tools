import hailtop.batch as hb
from shlex import quote
import yaml
from urllib.parse import urlparse

# runtime parameters
model = 'cpcnn' # cpcnn or dino4cells
plate_pate = 'gs://' # parent folder of plate folders
plates = ['BR0001', 'BR0002'] # plates to process, these should be subfolders of plate_path
output_folder = 'gs://' # folder in which output files will be placed
channel_names = 'DNA,Mito,...' # comma separated names of channels
channel_substrings = '-ch1,-ch2' # comma separated substrings of filename to identify channels
centers_path = 'gs://.../cellpose_{plate}.tsv' # path to cell centers, {plate} will be replaced with the plate name
averages = True # Whether to compute embedding averages per image

plate_path = urlparse(plate_path)
bucket_name = plate_path.netloc
input_folder = plate_path.path.rstrip('/')
output_folder = output_folder.rstrip('/')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

backend = hb.ServiceBackend(billing_project=config['hail-batch']['billing-project'],
                            remote_tmpdir=config['hail-batch']['remote-tmpdir'],
                            regions=config['hail-batch']['regions'])

b = hb.Batch(backend=backend, name=f'embedding {model}')
for plate in plates:
    j = b.new_job(name=f'embedding {model} {plate}')
    j.cloudfuse(bucket_name, '/images')
    j._machine_type = config[model]['machine-type']
    j.storage('30Gi') # should be large enough for pixi (12 GB), model and tsv output (not for images)
    
    model_weights = b.read_input(config[model]['model-weights'])
    centers_file = b.read_input(centers_path.format(plate=plate))
    
    num_workers = config[model]['num-workers']
    image_folder = f'{input_folder}/{plate}/'

    j.command('apt update')
    j.command('apt install -y git curl moreutils')
    j.command('git clone -b dev --single-branch https://github.com/atgu/microscopy_computational_tools.git')
    j.command('cd microscopy_computational_tools')
    j.command('curl -fsSL https://pixi.sh/install.sh | sh')
    j.command('export PATH=/root/.pixi/bin:$PATH')
    j.command('pixi install')
    j.command(f'pixi run python cell_embedding.py {model} {model_weights} /images/{quote(image_folder)} {quote(channel_names)} {quote(channel_substrings)} {quote(centers_file)} {num_workers} embedding.h5 crops.png {averages}')
    j.command(f'mv embedding.h5 {j.ofile1}')
    j.command(f'mv crops.png {j.ofile2}')
    b.write_output(j.ofile1, f'{output_folder}/embedding_{model}_{plate}.h5')
    b.write_output(j.ofile2, f'{output_folder}/embedding_{model}_{plate}.png')
    if averages:
        j.command(f'mv embedding_avg.tsv {j.ofile3}')
        b.write_output(j.ofile3, f'{output_folder}/embedding_{model}_{plate}_avg.tsv')
b.run()
