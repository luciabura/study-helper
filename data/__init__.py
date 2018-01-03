import os

# Directory path
from utilities.read_write import read_file

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_TEXT = read_file(os.path.join(DATA_DIR, 'handmade/mihalcea_sum.data'))
