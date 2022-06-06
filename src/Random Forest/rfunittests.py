import unittest
from rfclass import RandomForestRegressor, RegressionTree
import numpy as np

class TestModel(unittest.TestCase):
    def test_normal_fit(self):
        #check that the expected normal case
        self.model = RandomForestRegressor()
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        self.model.fit(X, y)

    def test_Xoneimplicitfeature(self):
        #check that this case of an implicit number of features works
        self.model = RandomForestRegressor()
        X = np.random.rand(100)
        y = np.random.rand(100)

        self.model.fit(X, y)

    def test_Xtoomanydims(self):
        #check that X > 2 dimensions raises ValueError:
        self.model = RandomForestRegressor()
        X = np.random.rand(100, 20, 10)
        y = np.random.rand(100)
        with self.assertRaises(ValueError):
            self.model.fit(X, y)

    def test_ytoomanydims(self):
        #check that y > 1 dimensions raises ValueError:
        self.model = RandomForestRegressor()
        X = np.random.rand(100, 10)
        y = np.random.rand(100, 10)
        with self.assertRaises(ValueError):
            self.model.fit(X, y)

    def test_Xycompatible(self):
        #check that X and y need to be equal in sample dimension:
        self.model = RandomForestRegressor()
        X = np.random.rand(100, 10)
        y = np.random.rand(67)
        with self.assertRaises(ValueError):
            self.model.fit(X, y)

    def test_split_criterion_decreases_after_split(self):
        #test that the split criterion decreases after splitting, which is true in a strict mathematical sense
        #if this fails, the split criterion is probably calculated erroneously 
        self.model = RegressionTree()
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        self.model.fit(X, y)
        node = self.model.root
        def test_helper(node):
            if node.children is None:
                return None
            for child in node.children:
                assertGreaterEqual(node.split_criterion_value, node.child.split_criterion_value)
                test_helper(child)





if __name__ == '__main__':
    unittest.main()
