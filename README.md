# steel_kaggle
Kaggle competition to identify steel defects

# Setup
Install python packages  
`pip install -r requirements.txt`
or if you have a gpu  
`pip install -r requirements-gpu.txt`
https://www.kaggle.com/c/severstal-steel-defect-detection

Accept the terms and conditions for the competition
You'll also need to install and set up the Kaggle API
```
mkdir data
cd data
kaggle competitions download -c severstal-steel-defect-detection
mkdir train test
unzip train_images.zip -d train
unzip test_images.zip -d test
unzip train.csv.zip
rm *.zip
```
unzip the training and test data and pu them into folders called
train and test respectively



