import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import create_datasets_from_yaml

create_datasets_from_yaml()