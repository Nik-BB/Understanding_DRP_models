# Running BinaryET 

Note that example_data is defined create_dataset.py where setting it to True runs the model out of the box using a subset of the full data to demonstrate code functionality (by default it is set to False).  Setting to False runs the model with the full dataset but  the same instructions for data downloading as in main readme need to first be followed to get the data. The paths to the datasets then need to be set in create_dataset.py. And to GDSC2 in binary_truth.py
BinaryET can be run in main_run_model.py using three arguments models_to_run, split_type and epochs. For example run  BinaryET for 100 epochs in cancer blind testing with 
python source_code/main_run_model.py db_tf c_blind 100

run_muti_dbs.py runs BinaryCB (model using chemberta) with 
python source_code/run_muti_dbs.py
when running the cmd from the BinaryET_andBinaryCB dir. 
.The version of chemberta can be controlled using db_model_version in the config dict in run_muti_dbs.py  see chemberta on hugging face for a full list of models that they support. Again data paths are set in create_dataset.py. 

