import hailtop.batch as hb
from shlex import quote
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

backend = hb.ServiceBackend(billing_project=config['hail-batch']['billing-project'],
                            remote_tmpdir=config['hail-batch']['remote-tmpdir'],
                            regions=config['hail-batch']['regions'])

bucket_name     = '' # without gs://
input_folder    = '' # path within the bucket, without trailing slash
plates          = ['BR00135656__2022-08-31T19_43_09-Measurement1'] # names of folders in input_folder
DNA_channel     = '-ch5' # the script will look for files named gs://{bucket_name}/{input_folder}/{plate}/*{DNA_channel}*
output_folder   = 'gs://....' # the script will create a file cellpose_{plate}.tsv in this folder, without trailing slash

b = hb.Batch(backend=backend, name='cellpose')
for plate in plates:
    j = b.new_job(name=f'cellpose {plate}')
    j.cloudfuse(bucket_name, '/images')
    j._machine_type = config['cellpose']['machine-type']
    j.storage('30Gi') # should be large enough for pixi (12 GB) and for tsv output (not for images)

    if config['cellpose']['model'] is not None:
        cellpose_model = b.read_input(config['cellpose']['model'])
        cellpose_model_size = b.read_input(config['cellpose']['model-size'])
        j.command('mkdir -p ~/.cellpose/models/')
        j.command(f'cp {cellpose_model} ~/.cellpose/models/nucleitorch_0')
        j.command(f'cp {cellpose_model_size} ~/.cellpose/models/size_nucleitorch_0.npy')

    num_processes = config['cellpose']['num-processes']
    process_string = str(num_processes) + ' -- ' + ' '.join(map(str, range(num_processes)))
    image_folder = f'{input_folder}/{plate}/'

    j.command('apt update')
    j.command('apt install -y git curl moreutils')
    j.command('git clone https://github.com/atgu/microscopy_computational_tools.git')
    j.command('cd microscopy_computational_tools')
    j.command('curl -fsSL https://pixi.sh/install.sh | sh')
    j.command('export PATH=/root/.pixi/bin:$PATH')
    j.command('pixi install')
    j.command(f'parallel -j {num_processes} pixi run python cellpose/run_cellpose.py /images/{quote(image_folder)} {quote(DNA_channel)} {process_string}')
    j.command(f'echo -e "file\\ti\\tj" > {j.ofile}')
    j.command(f'cat cellpose_* >> {j.ofile}')
    b.write_output(j.ofile, f'{output_folder}/cellpose_{plate}.tsv')
b.run() 