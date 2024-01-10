# Evitando a criação de arquivos .pyc
import sys
sys.dont_write_bytecode = True
# pyEasyMLcli.py
import os
import click
import json

@click.group()
def cli():
    pass

@cli.command()
@click.argument('current_dir', default='.', type=click.Path(exists=True))
def start(current_dir):
    click.echo("Seting framework up...")
    # dir from where the script is being executed
    current_dir = os.path.abspath(current_dir)
    
    proj_name = input("Project name: ")
    
    # folders important to the framework
    models_dir = os.path.join(current_dir, proj_name, 'models')
    utils_dir = os.path.join(current_dir, proj_name, 'Utils')
    dataset_dir = os.path.join(current_dir, proj_name, 'dataset')
    normalization_model_dir = os.path.join(models_dir, 'normalizationModel')
    
    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(utils_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(normalization_model_dir, exist_ok=True)

    # config data that'll save the client's folder
    config_data = {
        'client': os.path.join(current_dir, proj_name)
    }
    config_file_path = os.path.join(current_dir, proj_name, 'configs.json')
    with open(config_file_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    # Create src folder
    src_dir = os.path.join(current_dir, proj_name, 'src')
    os.makedirs(src_dir, exist_ok=True)

    # Create main file
    main_file_path = os.path.join(src_dir, 'main.py')
    with open(main_file_path, 'w') as main_file:
        main_file.write("#  Main file entrypoint header: Don't change or delete\n")
        main_file.write("from pyEasyML.add_module import add_pyEasyML_module\n")
        main_file.write("add_pyEasyML_module()\n")
        main_file.write("import pyEasyML\n")
        main_file.write("from pyEasyML import Settings\n")
        main_file.write("import os\n")
        main_file.write("settings = Settings(os.path.abspath(__file__))\n")

    click.echo("framework set up successfully!")

@cli.command()
def clean_normalization_models():
    current_dir = os.getcwd()
    
    click.echo("Cleaning normalization model...")
    normalization_model_dir = os.path.join(current_dir, 'models', 'normalizationModel')
    for file in os.listdir(normalization_model_dir):
        file_path = os.path.join(normalization_model_dir, file)
        os.remove(file_path)
    click.echo("Normalization model cleaned successfully!")

@cli.command()
def clean_models():
    current_dir = os.getcwd()
    
    click.echo("Cleaning models...")
    models_dir = os.path.join(current_dir, 'models')
    for file in os.listdir(models_dir):
        file_path = os.path.join(models_dir, file)
        if os.path.isfile(file_path) and file != 'normalizationModel':
            os.remove(file_path)
    click.echo("Models cleaned successfully!")

if __name__ == '__main__':
    cli()
