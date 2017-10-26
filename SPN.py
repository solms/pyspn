from NodeType import NodeType
from Node import SumNode, ProdNode, LeafNode
import numpy as np
import pandas as pd
from sklearn.mixture.gaussian_mixture import GaussianMixture
import random


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

        else:
            # Randomly decide whether to split on columns (for now)
            split_features = random.randint(0, 2) > 1
            print('split_features:', split_features)

            # Create leaf node if only 1x feature
            if len(data.columns) == 1:  # Create leaf node; scope == 1
                print('Creating leaf node from data with shape: ', data.shape)
                count = len([l for l in self.nodes if l.type == NodeType.LEAF])
                name = 'LEAF_{}'.format(count)  # Iteratively name leaf nodes
                node = LeafNode(name)
                parent.add_child(node)
                self.nodes.append(node)

            # Split features
            elif split_features:
                print('Creating product node from data with shape: ', data.shape)
                count = len([p for p in self.nodes if p.type == NodeType.PRODUCT])
                name = 'P{}'.format(count)  # Iteratively name product nodes
                node = ProdNode(name)
                if weight:  # Parent is SumNode
                    parent.add_child(node, weight)
                else:  # Parent is ProdNode
                    parent.add_child(node)
                self.nodes.append(node)
                transposed = data.T
                model = find_best_model(transposed)  # Find best Gaussian Mixture Model
                clusters = model.predict(transposed)  # Find the best clusters to split data into, row-wise
                classes = np.unique(clusters)
                if len(classes) == 1:
                    print('Classes don\'t want to split, forcing the issue.')
                    clusters = np.array([1, 2])
                    classes = np.unique(clusters)
                # Create the data subsets that will be children to this node
                for c in classes:
                    subset = transposed[clusters == c]
                    self.learn_spn(subset.T, node, None)

            # Split rows
            else:
                count = len([s for s in self.nodes if s.type == NodeType.SUM])
                name = 'S{}'.format(count)  # Iteratively name sum nodes
                print('Creating sum node,', name, ', from data with shape: ', data.shape)

                node = SumNode(name)
                if weight:  # Parent is SumNode
                    parent.add_child(node, weight)
                else:  # Parent is ProdNode
                    parent.add_child(node)
                self.nodes.append(node)
                model = find_best_model(data)  # Find best Gaussian Mixture Model
                clusters = model.predict(data)  # Find the best clusters to split data into, row-wise
                classes = np.unique(clusters)
                print('classes:', classes)
                # Create the data subsets that will be children to this node
                for c in classes:
                    subset = data[clusters == c]
                    weight = len(subset) / len(classes)
                    self.learn_spn(subset, node, weight)

    def __str__(self):
        text = ''
        for node in self.nodes:
            parents = [parent.name for parent in node.parents]
            text += node.name + ' (parents: ' + str(parents) + '), '
        return text


def find_best_model(data):
    """Tries to find the best GMM for the data"""
    lowest_bic = np.infty
    bic = []
    num_samples = len(data)
    upper = 10 if 10 < num_samples else num_samples
    n_components_range = range(1, upper + 1)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(data)
            bic.append(gmm.bic(data))
            if bic[-1] < lowest_bic and gmm.n_components > 1:  # Force a split into at least 2 components
                lowest_bic = bic[-1]
                best_gmm = gmm

    print('best_gmm.n_components:', best_gmm.n_components)

    return best_gmm

