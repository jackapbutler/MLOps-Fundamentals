import zipfile
with zipfile.ZipFile('./data/pima-indians-diabetes-database.zip','r') as zip_ref:
    zip_ref.extractall("./data/")
