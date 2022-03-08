#!/usr/bin/env python
# coding: utf-8

if __name__ == '__main__': #so pathos does not throw a fit
    import numpy as np
    from rfclass import RandomForestRegressor
    from rfbenchmark import benchmark_datasets, benchmark_params, plot_benchmark
    import sklearn.ensemble
    from sklearn.datasets import fetch_california_housing, load_diabetes, load_boston
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    datasets = [fetch_california_housing(), load_diabetes(), load_boston()]

    dataset = datasets[1]
    X, y = dataset['data'], dataset['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = RandomForestRegressor(n_estimators = 10)
    model.fit(X_train, y_train)

    plt.plot(model.get_feature_importances())

    model = sklearn.ensemble.RandomForestRegressor(n_estimators = 10)
    model.fit(X_train, y_train)

    plt.plot(model.feature_importances_)
    plt.show()

    #xvalues, myruntimes, myerrors = benchmark_datasets(RandomForestRegressor, datasets)
    #xvalues, skruntimes, skerrors = benchmark_datasets(sklearn.ensemble.RandomForestRegressor, datasets)

    #plot_benchmark(xvalues, myruntimes, myerrors, skruntimes, skerrors, xlabel = '#dataset')
    
    dataset = datasets[1]

    params = ['ccp_alpha', 'n_estimators']
    ranges = [np.linspace(0, 200.0, 10), range(1,10)]
    for param_name, value_range in zip(params, ranges):
        xvalues, myruntimes, myerrors = benchmark_params(RandomForestRegressor, dataset, param_name, value_range)
        xvalues, skruntimes, skerrors = benchmark_params(sklearn.ensemble.RandomForestRegressor, dataset, param_name, value_range)

        plot_benchmark(xvalues, myruntimes, myerrors, skruntimes, skerrors, xlabel = param_name)
