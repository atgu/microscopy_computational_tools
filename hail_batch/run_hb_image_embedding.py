import hailtop.batch as hb
from shlex import quote
import yaml
from urllib.parse import urlparse

# runtime parameters
model              = 'cellpose' # cellpose or unidino
plate_path         = 'gs://' # parent folder of plate folders
plates             = ['BR0001', 'BR0002'] # plates to process, these should be subfolders of plate_path
channel_names      = 'DNA,Mito,...' # comma separated names of channels used by the model; cellpose only uses DNA
channel_substrings = '-ch1,-ch2' # comma separated substrings of filename to identify channels
output_folder      = 'gs://....' # the script will create a file cellpose_{plate}.tsv in this folder

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
    j.storage('30Gi') # should be large enough for pixi (12 GB) and for tsv output (not for images)

    if model == 'cellpose' and config[model]['model-weights'] is not None:
        cellpose_model = b.read_input(config[model]['model-weights'])
        cellpose_model_size = b.read_input(config[model]['model-size'])
        j.command('mkdir -p ~/.cellpose/models/')
        j.command(f'cp {cellpose_model} ~/.cellpose/models/nucleitorch_0')
        j.command(f'cp {cellpose_model_size} ~/.cellpose/models/size_nucleitorch_0.npy')
        model_path = 'not_used'
    else:
        model_path = b.read_input(config[model]['model-weights'])

    num_processes = config[model]['num-processes']
    num_workers = 0
    process_string = str(num_processes) + ' -- ' + ' '.join(map(str, range(num_processes)))
    image_folder = f'{input_folder}/{plate}/'

    j.command('apt update')
    j.command('apt install -y git curl moreutils')
    j.command('git clone https://github.com/atgu/microscopy_computational_tools.git')
    j.command('cd microscopy_computational_tools')
    j.command('curl -fsSL https://pixi.sh/install.sh | sh')
    j.command('export PATH=/root/.pixi/bin:$PATH')
    j.command('pixi install')
    j.command(f'parallel -j {num_processes} pixi run python image_embedding.py {model} {model_path} /images/{quote(image_folder)} {quote(channel_names)} {quote(channel_substrings)} {num_workers} {process_string}')
    j.command(f'gzip -c embedding*.tsv >> {j.ofile}')
    b.write_output(j.ofile, f'{output_folder}/cellpose_{plate}.tsv.gz')
b.run() 