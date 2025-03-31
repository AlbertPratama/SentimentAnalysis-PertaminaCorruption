import pandas as pd
import numpy as np

df = pd.read_csv('./datasets/all_comments_temp.csv')
print(df.shape)

random_sample = df.sample(n=378, random_state=42)

random_sample.to_csv('sample_data_for_annotating.csv', index=False)