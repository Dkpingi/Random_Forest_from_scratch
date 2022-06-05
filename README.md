
# Random Forest from Scratch
This project tries to recreate the random forest regressor model of sklearn as a programming exercise. 
The created model is then benchmarked together with the standard regressor from sklearn to see how succeessful the implementation was in comparison.

# Benchmarks

# Datasets
First we compare the base performance of both models with three datasets from the sklearn library, namely the california housing market dataset, the diabetes dataset and the boston house price dataset:
![This is an image](figures/numDataset_vs_error.png)
![This is an image](figures/numDataset_vs_runtime.png)

The sklearn implementation tends to have lower train and test errors, except for one small exception.
Training times are similar for small datasets, but for the first dataset which is much larger my model is much slower.
This is probably due to a lack of parallelization of my model.

# Finetuning Parameters
We test the behaviour of both the train and validation error as well as the runtime of the models given different model finetuning parameters, while keeping all else equal:

## ccp_alpha
![This is an image](figures/ccp_alpha_vs_error.png)
![This is an image](figures/ccp_alpha_vs_runtime.png)

cost-complexity pruning prunes the Trees in order to reduce complexity, which hopefully leads to less overfitting.
My model doesn't seem to benefit very much from the pruning, however it also fits a lot less to the training set than the sklearn model, so of course there won't be a benefit.
Sklearn's ccp increases training error by a lot, while very slightly increasing validation performance.
Runtime is unaffected for sklearn, while my model has lower training time for aggressive pruning, which doesn't make any sense to me.

## min_samples_leaf
![This is an image](figures/min_samples_leaf_vs_error.png)
![This is an image](figures/min_samples_leaf_vs_runtime.png)

min_samples_leaf specifies that leaves of the individual decision trees cannot have less than min_samples_leaf amount of leaves. 
This is supposed to reduce overfitting, because the model cannot fit to every single training example exactly.

My model just gets worse in both training and validation error, while sklearns model gets a slight boost to validation performance up to a point, until it reverses again.
Runtime is unaffected for sklearn, while runtime goes to zero for my model after a certain point. My model's is actually the expected behaviour, 
so I suspect sklearn has safeguards against underfitting/not fitting at all.

## min_samples_split
![This is an image](figures/min_samples_split_vs_error.png)
![This is an image](figures/min_samples_split_vs_runtime.png)

min_samples_split determines that nodes below min_samples_split are not split any further and therefore become leaves of the tree. This has a similar effect to min_samples_leaf.
Unsurprisingly, the graphs look basically the same for the two models.

## n_estimators
![This is an image](figures/n_estimators_vs_error.png)
![This is an image](figures/n_estimators_vs_runtime.png)

n_estimators refers to the number of trees that are averaged over in the random forest model to build a random forest regressor.
The number of estimators reduces train and validation error for both models, but sklearn doesn't have much of a penalty for increasing estimators, while my model's runtime explodes.
This is very likely due to lack of parallelization on the part of my model, while sklearn naturally does have parallelization.

# Determining feature importances:
![This is an image](figures/feature_importances.png)

The feature importances determined by Sklearns model are roughly similar than the one determined by my model in that the two most important features are the same, 
but apart from that the feature importances are mostly different.


# Conclusion:
My implementation produces a rough approximation of sklearns implementation, but it runs much slower and has lower training and validation errors. 
The former is clearly due to the lack of parallelization in my implementation, the latter is more complicated, but sklearn's implementation is of course much more sophisticated so it's not surprising.