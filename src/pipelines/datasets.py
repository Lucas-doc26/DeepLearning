import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import pandas as pd

path = '/workspace/datasets/kyoto'

images = [os.path.join(path, x) for x in os.listdir(path) if x.lower().endswith(('.jpg', '.png', '.jpeg'))]
images.sort()  

print(len(images))

df = pd.DataFrame(images, columns=['path_image'])
os.makedirs('/workspace/CSV/kyoto', exist_ok=True)

df.to_csv('/workspace/CSV/kyoto/kyoto.csv', index=False)

df_train = df.iloc[:50]
df_val   = df.iloc[50:54]
df_test  = df.iloc[54:62]  


#df_train.to_csv('/home/lucas/DeepLearning/CSV/kyoto_train.csv', index=False)
#df_val.to_csv('/home/lucas/DeepLearning/CSV/kyoto_valid.csv', index=False)
#df_test.to_csv('/home/lucas/DeepLearning/CSV/kyoto_test.csv', index=False)

