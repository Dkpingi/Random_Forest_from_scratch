
# Random Forest from Scratch
This project tries to recreate the random forest regressor model of sklearn as a programming exercise. 
The created model is then benchmarked together with the standard regressor from sklearn to see how succeessful the implementation was in comparison.
As a final point, I want to analyze how sklearns implementation differs from mine and how that results in the benchmark differences.

# Benchmarks

# Datasets
First we compare the base performance of both models with three datasets:
![This is an image](figures/numDataset_error.png)
![This is an image](figures/numDataset_runtime.png)

# Parameters
We test the behaviour of both the train and validation error as well as the runtime of the models given different model finetuning parameters, while keeping all else equal:

## ccp_alpha
![This is an image](figures/ccp_alpha_vs_error.png)
![This is an image](figures/ccp_alpha_vs_runtime.png)

## min_samples_leaf
![This is an image](figures/min_samples_leaf_vs_error.png)
![This is an image](figures/min_samples_leaf_vs_runtime.png)

## min_samples_split
![This is an image](figures/min_samples_split_vs_error.png)
![This is an image](figures/min_samples_split_vs_runtime.png)

## n_estimators
![This is an image](figures/n_estimators_vs_error.png)
![This is an image](figures/n_estimators_vs_runtime.png)

## n_jobs
![This is an image](figures/n_jobs_vs_error.png)
![This is an image](figures/n_jobs_vs_runtime.png)

## Differences