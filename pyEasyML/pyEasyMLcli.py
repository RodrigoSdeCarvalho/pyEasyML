
# Evitando a criação de arquivos .pyc
import sys
sys.dont_write_bytecode = True
# pyEasyMLcli.py
import os
import re
import click
import json

@click.group()
def cli():
    pass

@cli.command()
def start():
    click.echo("Seting framework up...")
    script_dir = os.path.abspath(__file__)
    script_dir = re.sub(pattern="pyEasyML.*", repl = "pyEasyML/", string = script_dir)
    script_dir = os.path.abspath(script_dir)
    
    # dir from where the script is being executed
    current_dir = os.getcwd()
    
    # folders important to the framework
    models_dir = os.path.join(current_dir, 'models')
    utils_dir = os.path.join(current_dir, 'Utils')
    dataset_dir = os.path.join(current_dir, 'dataset')
    normalization_model_dir = os.path.join(models_dir, 'normalizationModel')
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(utils_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(normalization_model_dir, exist_ok=True)

    # config data that'll save the client's folder
    config_data = {
        'client': current_dir
    }
    config_file_path = os.path.join(script_dir, 'Configs', 'configs.json')
    with open(config_file_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    click.echo("framework set up successfully!")
    
@cli.command()
def clean_normalization_models():
    click.echo("Cleaning normalization model...")
    normalization_model_dir = os.path.join(script_dir, 'models', 'normalizationModel')
    for file in os.listdir(normalization_model_dir):
        file_path = os.path.join(normalization_model_dir, file)
        os.remove(file_path)
    click.echo("Normalization model cleaned successfully!")
    
@cli.command()
def clean_models():
    click.echo("Cleaning models...")
    models_dir = os.path.join(script_dir, 'models')
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        if os.path.isfile(file_path) and file != 'normalizationModel':
            os.remove(file_path)
    click.echo("Models cleaned successfully!")

if __name__ == '__main__':
    cli()
