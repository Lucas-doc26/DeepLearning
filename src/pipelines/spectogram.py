import tensorflow as tf
import pandas as pd
import os

csv_dir = '/datasets/audio/archive/spec_dataset/specs_128'
images = os.listdir(csv_dir)

full_image_path = [os.path.join(csv_dir, img) for img in images]

data = pd.DataFrame(full_image_path, columns=['path_image'])

train = data[:1000]
valid = data[1000:1065]
test = data[1065:]


os.makedirs("/home/lucas/DeepLearning/CSV/audio", exist_ok=True)
valid.to_csv("/home/lucas/DeepLearning/CSV/audio/audio_valid.csv", index=False)
train.to_csv("/home/lucas/DeepLearning/CSV/audio/audio_train.csv", index=False)
test.to_csv("/home/lucas/DeepLearning/CSV/audio/audio_test.csv", index=False)