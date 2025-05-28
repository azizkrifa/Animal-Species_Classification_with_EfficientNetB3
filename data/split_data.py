import splitfolders
from load_data import load_data

def split_data(): # split the dataset into train, validation, and test sets

  path=load_data() # load the dataset from Kaggle using the load_data function

  splitfolders.ratio(
      input=path,
      output="data/split_animals10",
      seed=1337,
      ratio=(.7, .2, .1)  # 70% train, 20% val, 10% test
  )

