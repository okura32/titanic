# %%
import numpy as np
import pandas as pd
from IPython.display import display

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
display(train.head())
display(test.head())
