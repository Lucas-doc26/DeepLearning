import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import SkipAutoencoderGenerator

skip = SkipAutoencoderGenerator(
    input_shape=(128,128,3),
    min_layers=2,
    max_layers=4,
    filters_list=[8,16,32,64,128],
    model_name="skip_autoencoder_random"
)   


skip.summary()