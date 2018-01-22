# Data augmentation tool


# Run script
First, install required libraries:

`pip install joblib textblob -U`

Download the necessary NLTK corpora

`python -m textblob.download_corpora`

Run the script

`python extend_dataset.py train.csv`

In the finish, there will be new csv files in the extended_data directory.
