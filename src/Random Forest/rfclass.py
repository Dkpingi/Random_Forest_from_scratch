
import numpy as np


class Node():
    '''
    Node class for Decision Tree implementation.
    '''
    def __init__(self, X, y, ids = None, depth = 0, max_depth = 5, criterion = 'squared_error', max_features = None, min_samples_split = 2, min_samples_leaf = 1, ccp_alpha = 0.0):
        #full dataset if no ids passed
        n_features = X.shape[1]
        if isinstance(max_features, int):
            self.max_features = max_features
        elif isinstance(max_features, float):
            self.max_features = int(max_features*n_features)
        elif max_features == 'auto' or max_features is None:
            self.max_features = n_features
        elif max_features == 'sqrt':
            self.max_features = int(np.sqrt(n_features))
        elif max_features == 'log2':
            self.max_features = int(np.log2(n_features))
        else:
            raise ValueError('max_features not in [int, float, "auto", "sqrt", "log2"]')
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
        self.feature_importances_ = None
   
        
        abs_err = lambda y, i : np.sum(np.abs((y[i] - np.median(y[i]))))
        mse     = lambda y, i : np.sum((y[i] - np.mean(y[i]))**2)
        var_red = lambda y, i : np.sum((y[i][:, None] - y[i][None, :])**2)
        poisson = lambda y, i : np.sum(y[i]*np.log(y[i]/np.mean(y[i])) - y[i] + np.mean(y[i]))
        
        self.crit = criterion
        criteria = {'squared_error' : mse, 'variance_reduction' : var_red, 
                    'absolute_error' : abs_err, 'poisson' : poisson}
        self.criterion = criteria[criterion]
        
        self.min_samples_split = max(2, min_samples_split)
        self.min_samples_leaf = max(1, min_samples_leaf)
        self.ccp_alpha = ccp_alpha

    def build_tree(self, X, y):
        '''
        Order samples by feature, test all splits along that feature. Repeat for all features included. Take best split.
        '''
        if len(self.ids) < self.min_samples_split:    
            self.feature_importances_ = np.zeros(X.shape[1])
            return self.feature_importances_
        
        l_ids = []
        r_ids = []
        
        self.n_features = X.shape[1]
        self.split_criterion_value = np.inf
        self.split_value = None
        self.split_features = np.random.choice(list(range(X.shape[1])), size = self.max_features, replace = False)


        for split_feature in self.split_features:
            sorted_ids = np.argsort(X[self.ids, split_feature])
            sorted_ids = np.take_along_axis(self.ids, sorted_ids, axis = 0)
            
            for i in range(self.min_samples_leaf, len(sorted_ids) - self.min_samples_leaf + 1):
                l_ids_temp = sorted_ids[:i]
                r_ids_temp = sorted_ids[i:]

                criterion_temp = self.calc_criterion(y, self.ids, l_ids_temp, r_ids_temp)

                if self.split_criterion_value > criterion_temp:
                    self.split_criterion_value = criterion_temp
                    self.split_value = 0.5*(X[l_ids_temp[-1], split_feature] + X[r_ids_temp[0], split_feature])
                    self.split_feature = split_feature
                    l_ids = l_ids_temp
                    r_ids = r_ids_temp

        self.feature_importances_ = np.zeros(X.shape[1])
        if self.split_criterion_value != np.inf:
            self.feature_importances_[self.split_feature] += self.criterion(y, self.ids) - self.split_criterion_value
            self.children.append(Node(X, y, ids = l_ids, criterion = self.crit, 
                                      max_depth = self.max_depth, depth = self.depth + 1, max_features = self.max_features,
                                      min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf, ccp_alpha = self.ccp_alpha))
            self.children.append(Node(X, y, ids = r_ids, criterion = self.crit,
                                      max_depth = self.max_depth, depth = self.depth + 1, max_features = self.max_features,
                                      min_samples_split = self.min_samples_split, min_samples_leaf = self.min_samples_leaf, ccp_alpha = self.ccp_alpha))
            
            if self.depth < self.max_depth:
                for child in self.children:
                    self.feature_importances_ += child.build_tree(X, y)

        return self.feature_importances_
                    
    
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
        Calculate loss function of y after splitting ids into l_ids and r_ids.
        '''

        split_criterion_value = (len(l_ids)/len(ids))*self.criterion(y, l_ids) + (len(r_ids)/len(ids))*self.criterion(y, r_ids)
        return split_criterion_value
        
    def ccp_prune(self, y):
        if not len(self.children) == 0:
            branch_err = self.criterion(y, self.ids)
            leaf_err, N = self.calc_leaf_err(y)
            alpha_eff = (branch_err - leaf_err)/(N - 1)
            if alpha_eff < self.ccp_alpha:
                self.children = []
            else:
                for child in self.children:
                    child.ccp_prune(y)
            
    def calc_leaf_err(self, y):
        err = 0.0
        N = 0
        if not len(self.children) == 0:
            for child in self.children:
                err_, N_ = child.calc_leaf_err(y)
                err += err_
                N += N_
        else:
            err = self.criterion(y, self.ids)
            N += 1

        return err, N

class RegressionTree():
    def __init__(self, max_depth = 100, criterion = 'squared_error', ccp_alpha = 0.0, min_samples_split = 2, min_samples_leaf = 1, max_features = 1.0):
        self.root = None
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
             
    def fit(self, X, y):
        self.root = Node(X, y, max_depth = self.max_depth, criterion = self.criterion, max_features = self.max_features, min_samples_split = self.min_samples_split,
                         min_samples_leaf = self.min_samples_leaf,  ccp_alpha = self.ccp_alpha)
        self.root.build_tree(X, y)
        if self.ccp_alpha > 0.0:
            self.root.ccp_prune(y)
        return self
        
    def predict(self, X):
        return self.root.evaluate_node(X)

    def return_self(self):
        return self

    def get_feature_importances(self):
        return self.root.feature_importances_
    

    
class RandomForestRegressor():
    def __init__(self, n_estimators = 10, max_depth = 100, criterion = 'squared_error', ccp_alpha = 0.0, max_features = 1.0, min_samples_split = 2, min_samples_leaf = 1):
        self.max_features = max_features
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(RegressionTree(max_depth = max_depth, criterion = criterion, ccp_alpha = ccp_alpha,  
                                             min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, 
                                             max_features = self.max_features))
        self.n_features = None

    def fit(self, X, y):
        '''
        Fit each tree with sampled features and bootstrap samples.
        '''
        
        self.X = X
        self.y = y

        if self.X.ndim == 1:
            X = X[:, None]
        
        self.n_features = X.shape[1]

        if self.X.ndim > 2:
            raise ValueError('X should have dims [n_samples, n_features]')

        if self.y.ndim > 1:
            raise ValueError('y should have dims [n_samples]')

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError('X and y should have same size in first dimension.')
        
        n_samples = [np.random.choice(list(range(X.shape[0])), size = int(X.shape[0]/10.0)) for _ in self.trees]
        for tree, samples in zip(self.trees, n_samples):
            tree.fit(X[samples], y[samples])

        self.feature_importances_ = self.get_feature_importances()
    
     
    def fit_helper(self, tree, ids):
        return tree.fit(self.X[ids], self.y[ids])

    def predict(self, X):
        '''
        Average over all tree predictions.
        '''
        prediction = np.mean([tree.predict(X) for tree in self.trees], axis = 0)
        return prediction
    
    def get_feature_importances(self):
        '''
        Get feature importances by summing over all variance reductions at each node for each feature,
        normalized by the number of times the feature is used for the trees in the forest.
        '''
        #Get importances from each tree
        
        importance_arr = [tree.get_feature_importances() for tree in self.trees]

        importances = np.zeros(self.n_features)
        for imp in importance_arr:
            importances += imp
        importances /= np.sum(importances + 1e-16)
        return importances