from NodeType import NodeType
from Node import SumNode, ProdNode, LeafNode
import numpy as np
import pandas as pd
from sklearn.mixture.gaussian_mixture import GaussianMixture
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools


class SPN:
    """The structure for an SPN"""
    def __init__(self, nodes=[]):
        self.nodes = nodes
        self.leaves = {}
        for node in nodes:
            if node.type == NodeType.LEAF:
                self.leaves[node.name] = node

    def update_leaf(self, name, value):
        """Update the value of the leaf"""
        self.leaves[name].value = value

    def get_root(self):
        for node in self.nodes:
            if len(node.parents) == 0:
                return node

    def get_root_value(self, max_mode=False):
        """Calculate the value at the root, calculated bottom-up"""
        return self.get_root().get_value(max_mode=max_mode)

    def calculate_map_route_counts(self):
        """Calculate the routes followed for MAP state, and update counts"""
        self.get_root().update_map_weight_counts()

    def normalise_counts_as_weights(self, node=None):
        """Traverse the tree, and for sum nodes, normalise counts on links to sum to one and set as weights"""
        if not node:
            root = self.get_root()
            root.normalise_counts_as_weights()
            self.normalise_counts_as_weights(root)
        else:
            for child in node.children:
                if child.type == NodeType.SUM:
                    child.normalise_counts_as_weights()
                self.normalise_counts_as_weights(child)

    def fit(self, variables, data, epochs=100):
        """Tries to lean appropriate weights for the current structure by applying hard EM"""
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs))
            for leaf in self.leaves.values():
                leaf.value = 0
            for sample in data:
                for i in range(len(sample)):
                    # Set the leaves to match training example
                    if sample[i] == 1:
                        self.leaves[variables[i]].value = 1.0
                        self.leaves[variables[i]+'_'].value = 0.0
                    else :
                        self.leaves[variables[i]].value = 0.0
                        self.leaves[variables[i] + '_'].value = 1.0
                # Calculate the max version bottom-up
                self.get_root_value(True)
                # Update the counts top-down
                self.calculate_map_route_counts()
            self.normalise_counts_as_weights()
        for link in self.get_root().links.values():
            print(link)

    def create_structure(self, data, variables):
        """Tries to learn appropriate structure from the data"""
        self.learn_spn(pd.DataFrame(data, columns=variables))

    def learn_spn(self, data, parent=None, weight=None):
        # Split rows on first pass
        if not parent:
            print('Creating root node from data with shape: ', data.shape)
            root = SumNode('root')
            self.nodes.append(root)
            model = find_best_model(data)  # Find best Gaussian Mixture Model
            clusters = model.predict(data)  # Find the best clusters to split data into, row-wise
            classes = np.unique(clusters)
            # Create the data subsets that will be children to this node
            for c in classes:
                subset = data[clusters == c]
                weight = len(subset) / len(classes)
                self.learn_spn(subset, root, weight)

        # Parent is SumNode, so split column-wise
        elif weight:
            print('Creating product node from data with shape: ', data.shape)
            count = len([p for p in self.nodes if p.type == NodeType.PRODUCT])
            name = 'P{}'.format(count)  # Iteratively name product nodes
            node = ProdNode(name)
            parent.add_child(node, weight)
            self.nodes.append(node)
            transposed = data.T
            model = find_best_model(transposed)  # Find best Gaussian Mixture Model
            clusters = model.predict(transposed)  # Find the best clusters to split data into, row-wise
            classes = np.unique(clusters)
            # Create the data subsets that will be children to this node
            for c in classes:
                subset = transposed[clusters == c]
                self.learn_spn(subset.T, node, None)

        # Parent is ProdNode, so split row-wise
        else:
            if len(data.columns) == 1:  # Create leaf node; scope == 1
                print('Creating leaf node from data with shape: ', data.shape)
            else:
                print('Creating sum node from data with shape: ', data.shape)


def find_best_model(data):
    """Tries to find the best GMM for the data"""
    lowest_bic = np.infty
    bic = []
    num_samples = len(data)
    upper = 10 if 10 < num_samples else num_samples
    n_components_range = range(1, upper)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    return best_gmm

