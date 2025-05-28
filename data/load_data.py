import kagglehub  #This script downloads the Animals-10 dataset from Kaggle using the kagglehub library.
def load_data():
    path = kagglehub.dataset_download("alessiocorrado99/animals10")
    return path 
