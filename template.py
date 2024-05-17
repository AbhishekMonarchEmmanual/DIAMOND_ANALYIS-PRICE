from pathlib import Path
import os

# Define the source directory
src = input(str("GIVE YUR PROJECT NAME HERE : "))

# List of files and directories to create
list_of_files = [
    os.path.join(".github", "workflows", "main.yml"),
    os.path.join(src, 'config', '__init__.py'),
    os.path.join(src, 'config', 'config.py'),
    os.path.join(src, 'config', 'artifacts.py'),
    os.path.join(src, 'components', '__init__.py'),
    os.path.join(src, 'components', "data_ingestion.py"),
    os.path.join(src, 'components', "data_transformation.py"),
    os.path.join(src, 'components', "model_trainer.py"),
    os.path.join(src, 'components', "model_evaluation.py"),
    os.path.join(src, 'pipeline', '__init__.py'),
    os.path.join(src, 'pipeline', 'training.py'),
    os.path.join(src, 'pipeline', 'predicition.py'),
    os.path.join(src, 'utils', '__init__.py'),
    os.path.join(src, 'utils', 'utils.py'),
    os.path.join(src, 'logger', '__init__.py'),
    os.path.join(src, 'logger', 'logging.py'),
    os.path.join(src, 'exception', '__init__.py'),
    os.path.join(src, 'exception', 'exception.py'),
    os.path.join('test', 'unit', '__init__.py'),
    os.path.join('test', 'integeration', '__init__.py'),
    os.path.join('init_setup.sh'),
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiments/experiments.ipynb",
    "main.py"
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file
