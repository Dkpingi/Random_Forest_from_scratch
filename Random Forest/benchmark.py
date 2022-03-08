
def benchmark_datasets(model_class, datasets):
    '''
    Benchmarks the model for the given dataset for a single parameter across multiple values.
    Uses default values for all other parameters. Returns both runtimes and predictive performances.
    '''  
    runtimes = []
    errors = {'train' : [], 'val' : []}
    for dataset in datasets:
        X, y = dataset['data'], dataset['target']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        t0 = time.time()
        model = model_class()
        model.fit(X_train, y_train)
        t1 = time.time()
        runtimes.append(t1 - t0)

        errors['train'].append(mape(model.predict(X_train), y_train))
        errors['val'].append(mape(model.predict(X_val), y_val))
    
    return runtimes, errors

def benchmark_params(model_class, dataset, param_name, param_values):
    '''
    Benchmarks the model for the given dataset for a single parameter across multiple values.
    Uses default values for all other parameters. Returns both runtimes and predictive performances.
    '''
    
    X, y = dataset['data'], dataset['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    runtimes = []
    errors = {'train' : [], 'val' : []}
    for param_value in param_values:
        param_dict = {param_name : param_value}
        t0 = time.time()
        model = model_class(**param_dict)
        model.fit(X_train, y_train)
        t1 = time.time()
        runtimes.append(t1 - t0)

        errors['train'].append(mape(model.predict(X_train), y_train))
        errors['val'].append(mape(model.predict(X_val), y_val))
    
    return runtimes, errors

def plot_benchmark(myruntimes, myerrors, skruntimes, skerrors):
    plt.title('Training Time')
    plt.plot(skruntimes, label = 'skmodel')
    plt.plot(myruntimes, label = 'mymodel')
    plt.legend()
    plt.show()

    plt.title('Train/Test Error')
    plt.plot(skerrors['train'], label = 'skmodeltrain')
    plt.plot(skerrors['val'], label = 'skmodeltest')
    plt.plot(myerrors['train'], label = 'mymodeltrain')
    plt.plot(myerrors['val'], label = 'mymodeltest')
    plt.legend()
    plt.show()