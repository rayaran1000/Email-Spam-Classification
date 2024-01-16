#Importing System Libraries
import os
import sys
from dataclasses import dataclass

#Importing Dataframe handling libraries
import numpy as np
import pandas as pd

#Importing logger and exception handlers
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

#Scikit Learn libraries and functions


@dataclass
class DataTransformationConfig:

    processor_file_path : str = os.path.join('artifacts','processor.pkl')

class DataTransformation:

