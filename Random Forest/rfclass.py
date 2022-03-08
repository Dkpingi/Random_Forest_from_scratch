
import numpy as np
from pathos.multiprocessing import ProcessingPool
import time 

class Node():
    '''
    Node class for Decision Tree implementation.
    '''
    def __init__(self, X, y, ids = None, depth = 0, max_depth = 5, criterion = 'squared_error', max_features = None):
        #full dataset if no ids passed
        if ids is not None:
            self.ids = ids
        else:
            self.ids = np.array(list(range(X.shape[0])))
        self.max_depth = max_depth
        self.importances = np.zeros(X.shape[1])   
        self.depth = depth
    
        self.value = np.mean(y[self.ids]) 
        
        #initialized later   
        self.considered_features = None
        self.split_feature = None
        self.split_value = None
        self.split_impurity_reduction = None
        self.children = []
        self.parent = None
        self.n_features = None
    
        self.max_features = max_features 
        
        abs_err = lambda y, i : np.sum(np.abs((y[i] - np.median(y[i]))))
        mse     = lambda y, i : np.sum((y[i] - np.mean(y[i]))**2)
        var_red = lambda y, i : np.sum((y[i][:, None] - y[i][None, :])**2)
        poisson = lambda y, i : np.sum(y[i]*np.log(y[i]/np.mean(y[i])) - y[i] + np.mean(y[i]))
        
        self.crit = criterion
        criteria = {'squared_error' : mse, 'variance_reduction' : var_red, 
                    'absolute_error' : abs_err, 'poisson' : poisson}
        self.criterion = criteria[criterion]
        
    def build_tree(self, X, y, min_samples_split = 2):
        min_samples_split = max(1, min_samples_split)
        if self.children != [] or len(self.ids) < min_samples_split:
            return 0
        
        l_ids = []
        r_ids = []
        
        self.n_features = X.shape[1]
        self.split_impurity_reduction = np.inf
        self.split_value = None
        self.split_features = np.random.choice(list(range(X.shape[1])), size = self.max_features(X.shape[1]))
        
        for criterion_id in self.split_features:
            sorted_ids = np.argsort(X[self.ids, criterion_id])
            sorted_ids = np.take_along_axis(self.ids, sorted_ids, axis = 0)
            
            for i in range(min_samples_split, len(sorted_ids) - min_samples_split):
                l_ids_temp = sorted_ids[:i]
                r_ids_temp = sorted_ids[i:]

                criterion_temp = self.calc_criterion(y, self.ids, l_ids_temp, r_ids_temp)
                if self.split_impurity_reduction > criterion_temp:
                    self.split_impurity_reduction = criterion_temp
                    self.split_value = 0.5*(X[l_ids_temp[-1], criterion_id] + X[r_ids_temp[0], criterion_id])
                    self.split_feature = criterion_id
                    l_ids = l_ids_temp
                    r_ids = r_ids_temp
        
        if len(l_ids) > 0 and len(r_ids) > 0:
            self.children.append(Node(X, y, ids = l_ids, criterion = self.crit, 
                                      max_depth = self.max_depth, depth = self.depth + 1, max_features = self.max_features))
            self.children.append(Node(X, y, ids = r_ids, criterion = self.crit,
                                      max_depth = self.max_depth, depth = self.depth + 1, max_features = self.max_features))
            
            for child in self.children:
                child.parent = self
                
            if self.depth < self.max_depth:
                for child in self.children:
                    child.build_tree(X, y, min_samples_split = min_samples_split)
    
    def evaluate_node(self, X):
        y = []
        for x in X:
            node = self
            while len(node.children) != 0:
                if x[node.split_feature] <= node.split_value:
                    node = node.children[0]
                else:
                    node = node.children[1]
            y.append(node.value)
        return np.array(y)
        
    def calc_criterion(self, y, ids, l_ids, r_ids):
        '''
        Calculate reduction of impurity measure by splitting ids into l_ids and r_ids.
        '''

        variance_reduction = (len(l_ids)/len(ids))*self.criterion(y, l_ids) + (len(r_ids)/len(ids))*self.criterion(y, r_ids)
        return variance_reduction
    
    def get_feature_importances(self):
        if self.children != []:
            self.importances[self.split_feature] += self.split_impurity_reduction
            for child in self.children:
                self.importances += child.get_feature_importances()
            return self.importances
        else:
            return 0.0
        
    def ccp_prune(self, y, ccp_alpha):
        if not len(self.children) == 0:
            branch_err = self.criterion(y, self.ids)
            leaf_err, N = self.calc_leaf_err(y)
            alpha_eff = (branch_err - leaf_err)/(N - 1)
            print('branch', branch_err)
            print('leaf', leaf_err)
            print('alpha', alpha_eff)
            if alpha_eff < ccp_alpha:
                self.children = []
            else:
                for child in self.children:
                    child.ccp_prune(y, ccp_alpha)
            
    def calc_leaf_err(self, y):
        err = 0.0
        N = 0
        if not len(self.children) == 0:
            for child in self.children:
                err_, N_ = child.calc_leaf_err(y)
                err += err_
                N += N_
        else:
            err = self.criterion(y, self.ids) #probably incomplete
            N += 1

        return err, N
    
class RegressionTree():
    def __init__(self, max_depth, criterion, ccp_alpha, min_samples_split, max_features):
        self.root = None
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.max_features = max_features
             
    def fit(self, X, y):
        self.root = Node(X, y, max_depth = self.max_depth, criterion = self.criterion, max_features = self.max_features)
        self.root.build_tree(X, y, min_samples_split = self.min_samples_split)
        if not self.ccp_alpha == 0.0:
            self.root.ccp_prune(y, self.ccp_alpha)
        return self
        
    def predict(self, X):
        return self.root.evaluate_node(X)
    

    
class RandomForestRegressor():
    def __init__(self, n_estimators = 10, max_depth = 4, criterion = 'squared_error', ccp_alpha = 0.0, max_features = None, min_samples_split = 2):
        if isinstance(max_features, int):
            self.max_features = lambda n_features : max_features
        elif isinstance(max_features, float):
            self.max_features = lambda n_features : int(max_features*n_features)
        elif max_features == 'auto' or max_features is None:
            self.max_features = lambda n_features : n_features
        elif max_features == 'sqrt':
            self.max_features = lambda n_features : int(np.sqrt(n_features))
        elif max_features == 'log2':
            self.max_features = lambda n_features : int(np.log2(n_features))
        else:
            raise ValueError('max_features not in [int, float, "auto", "sqrt", "log2"]')
        
        self.trees = []
        for i in range(n_estimators):
            self.trees.append(RegressionTree(max_depth = max_depth, criterion = criterion, ccp_alpha = ccp_alpha,  
                                             min_samples_split = min_samples_split, max_features = self.max_features))
        self.n_features = None
    def fit(self, X, y):
        '''
        Fit each tree with sampled features and bootstrap samples.
        '''
        self.n_features = X.shape[1]
        
        pool = ProcessingPool(nodes=len(self.trees))
        results = []
        for tree in self.trees:
            n_samples = np.random.choice(list(range(X.shape[0])), size = int(X.shape[0]/10.0))
            
            X_fit = X[n_samples]
            y_fit = y[n_samples]
            results.append(pool.apipe(tree.fit, X_fit, y_fit))

        for result in results:
            while not result.ready():
                pass
                #time.sleep(0.1); print(".", end=' ')
        
        for i, tree in enumerate(self.trees):
            self.trees[i] = results[i].get()

                
    
    def predict(self, X):
        '''
        Average over all tree predictions.
        '''
        prediction = 0.0
        for idx, tree in enumerate(self.trees):
            prediction += tree.predict(X)
        prediction /= len(self.trees)
        return prediction
    
    def get_feature_importances(self):
        '''
        Get feature importances by summing over all variance reductions at each node for each feature,
        normalized by the number of times the feature is used for the trees in the forest.
        '''
        #Get importances from each tree
        importances = np.zeros(self.n_features)
        for idx, tree in enumerate(self.trees):
            importances += tree.root.get_feature_importances()       
        importances /= np.sum(importances)
        return importances