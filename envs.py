import os
import logging
import sys
from os.path import join
PROJECT_FOLDER = os.path.dirname(__file__)
print(PROJECT_FOLDER)
sys.path.append(join(PROJECT_FOLDER))

# Define the dataset folder and model folder based on environment
# HOME_DATA_FOLDER = '/ssd/HGN/data'
HOME_DATA_FOLDER = join(PROJECT_FOLDER, 'data')
KG_DATASET_FOLDER = join(PROJECT_FOLDER, 'data')
COMMON_KG_DATASET_FOLDER = join(PROJECT_FOLDER, 'commonsensedata')
CodEX_DATASET_FOLDER = join(PROJECT_FOLDER, 'codexdata')
OUTPUT_FOLDER = join(HOME_DATA_FOLDER, 'outputs')