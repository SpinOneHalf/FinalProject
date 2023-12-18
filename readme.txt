Review of Unet and Fine tuning training.
The main file for training U-net was simple_data_set.py
This makes a U-net object with random weights and trains of the on the labeled data

The Fine tuning happens on the hard_data_set.py
The script takes the predictive model, and U-net and trains on the label data set.
This happens by updating both networks at the same time.