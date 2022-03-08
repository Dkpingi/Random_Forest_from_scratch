from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

def benchmark_datasets(model_class, datasets, label):
    '''
    Benchmarks the model for the given dataset for a single parameter across multiple values.
    Uses default values for all other parameters. Returns both runtimes and predictive performances.
    '''  
    plot_dict = {'x_values' : [], 'runtime' : [], 'train_err' : [], 'val_err' : [], 'label' : label, 'xlabel' : '#Dataset'}
    for i, dataset in enumerate(datasets):
        X, y = dataset['data'], dataset['target']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        t0 = time.time()
        model = model_class()
        model.fit(X_train, y_train)
        t1 = time.time()

        plot_dict['runtime'].append(t1 - t0)
        plot_dict['x_values'].append(i)
        plot_dict['train_err'].append(mape(model.predict(X_train), y_train))
        plot_dict['val_err'].append(mape(model.predict(X_val), y_val))
    
    return plot_dict

def benchmark_params(model_class, dataset, param_name, param_values, label):
    '''
    Benchmarks the model for the given dataset for a single parameter across multiple values.
    Uses default values for all other parameters. Returns both runtimes and predictive performances.
    '''
    
    X, y = dataset['data'], dataset['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    xvalues = []
    runtimes = []
    plot_dict = {'x_values' : [], 'runtime' : [], 'train_err' : [], 'val_err' : [], 'label' : label, 'xlabel' : param_name}
    for param_value in param_values:
        param_dict = {param_name : param_value}
        t0 = time.time()
        model = model_class(**param_dict)
        model.fit(X_train, y_train)
        t1 = time.time()

        plot_dict['runtime'].append(t1 - t0)
        plot_dict['x_values'].append(param_value)
        plot_dict['train_err'].append(mape(model.predict(X_train), y_train))
        plot_dict['val_err'].append(mape(model.predict(X_val), y_val))
    
    return plot_dict

def plot_benchmark(plot_dict_list):
    for plot_dict in plot_dict_list:
        xvalues = plot_dict['x_values']
        runtime = plot_dict['runtime']
        train_err = plot_dict['train_err']
        val_err = plot_dict['val_err']
        label = plot_dict['label']
        xlabel = plot_dict['xlabel']


        plt.plot(xvalues, runtime, label = label)
        plt.title('Training Time')
        plt.xlabel(xlabel)
        plt.legend()

    plt.show()

    for plot_dict in plot_dict_list:
        xvalues = plot_dict['x_values']
        runtime = plot_dict['runtime']
        train_err = plot_dict['train_err']
        val_err = plot_dict['val_err']
        label = plot_dict['label']
        xlabel = plot_dict['xlabel']

        plt.plot(xvalues, train_err, label = f'{label}train')
        plt.title('Train/Test Error')
        plt.plot(xvalues, val_err, label = f'{label}val')
        plt.xlabel(xlabel)
        plt.legend()
    
    plt.show()