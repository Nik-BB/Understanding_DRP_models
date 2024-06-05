# DRP-aberrations-and-comparisons


Exploration of how  different datatypes and subnetworks of state-of-the-art drug response prediciton (DRP) models effect performance. 


## Using the repository

### Running the benchmarks 
Running the benchmarks that do not use omics or chemical structure data is done in marker_benchmark.py and mean_model_benchmark.py

marker_benchmark.py runs the marker benchmark while mean_model_benchmark.py runs the drug average and cell line average benchmarks. 

The code used to create these models can be found in source_code/models

If the datasets (instructions on downloading given below) are not in the same paths specified in the code, then these paths (omic_dir_path, gdsc2_target_path and pubchem_ids_path) need to be set in the code to where you have the datasets. 

### Running the literate models

main_run_model.py file can be run for each model and testing type by specific both as arguments.

The paths to the datasets may need to be set as described above

## Datasetses
The datasets needed to re-train the models are publically available.

* Transcriptomics and genomics data can be downloaded from the Genomics of drug sensitivity in cancer database https://www.cancerrxgene.org/

* Drug response data in the form of IC50 values can be downloaded form values from GDSC https://www.cancerrxgene.org/

PubChem ID's and smiles strings for the drugs in GDSC can be found from  https://pubchem.ncbi.nlm.nih.gov/

Once downloaded the path to these datasets needs to be set (see using the repository above).

### Processing data 

New train test splits can be genrated by using the train_test_split notebook

A dict mapping drug names to smiles strings can be found and saved using the create_drug_to_smiles_mapping_gdsc2 method from data_loading.py. This dict can then be read in, in later use. 

## Problem Formulation 

The goal of DRP is to predict how effective different drugs are for different cancer types. 
Here we predict the I50 values, the concentration of a drug needed to inhibit the activity of a cell lie by 50%, as a measure of efficacy. 
We feed omics profiles of cell lines and simple column representaions of drugs though a neural network to do this. 

Consider the traning set $T = \lbrace \boldsymbol{x_{c,i}}, \boldsymbol{x_{d,i}}, y_i \rbrace$  where $\boldsymbol{x_{c,i}}$, $\boldsymbol{x_{d,i}}$  are representation of the $i^{th}$ cell line and drug respectively and
 $y_i$ is the IC50 value associated with the $i^{th}$ cell line drug pair.

 Thus, we want to find a model, $M$, that takes $\boldsymbol{x_{c,i}}$ and $\boldsymbol{x_{d,i}}$ as inputs and predicts for the corresponding IC50 value $\hat{y_i}$ such that $M(\boldsymbol{x_{c,i}}, \boldsymbol{x_{d,i}})=\hat{y_i}$.