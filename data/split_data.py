!pip install split-folders
import splitfolders

def split_data():
  path=load_data()
  splitfolders.ratio(
      input=path,
      output="data/split_animals10",
      seed=1337,
      ratio=(.7, .2, .1)  # 70% train, 20% val, 10% test
  )

