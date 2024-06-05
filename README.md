# DRP-aberrations-and-comparisons


Exploration of how  different datatypes and subnetworks of state-of-the-art drug response prediciton (DRP) models effect performance. 


## Using the repository

The notebook train_test_split.ipynb needs to be first run to do the train test split. The file get_mol_graph.py needs to be run to find the molecular graph representation of the chemicals. 

Then the main_run_model.py file can be run for each model and testing type by specific both as arguments. 
To run our benchmarks, use the files mean_model_benchmark.py or marker_benchmark.py

## Problem Formulation 

The goal of DRP is to predict how effective different drugs are for different cancer types. 
Here we predict the I50 values, the concentration of a drug needed to inhibit the activity of a cell lie by 50%, as a measure of efficacy. 
We feed omics profiles of cell lines and simple column representaions of drugs though a neural network to do this. 

Consider the traning set $T = \lbrace \boldsymbol{x_{c,i}}, \boldsymbol{x_{d,i}}, y_i \rbrace$  where $\boldsymbol{x_{c,i}}$, $\boldsymbol{x_{d,i}}$  are representation of the $i^{th}$ cell line and drug respectively and
 $y_i$ is the IC50 value associated with the $i^{th}$ cell line drug pair.

 Thus, we want to find a model, $M$, that takes $\boldsymbol{x_{c,i}}$ and $\boldsymbol{x_{d,i}}$ as inputs and predicts for the corresponding IC50 value $\hat{y_i}$ such that $M(\boldsymbol{x_{c,i}}, \boldsymbol{x_{d,i}})=\hat{y_i}$.


## Datasetses
The datasets needed to re-train the models are publically available.
 Please see the paper to find the  and downloaded datasets needed. 

Once downloaded the path to these datasets needs to be set. 
