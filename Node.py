from NodeType import NodeType
import numpy as np
import random


class Node(object):
    """An SPN node"""
    def add_parent(self, parent):
        self.parents.append(parent)

    def add_child(self, child):
        self.children.append(child)

    def update_map_weight_counts(self):
        raise NotImplementedError()

    def get_value(self, max_mode=False):
        raise NotImplementedError()

    def __str__(self):
        return '{type} node {name}' \
            .format(type=self.type.name, name=self.name)


class SumNode(Node):
    """An SPN sum node"""
    def __init__(self, name, children):
        self.name = name
        self.children = children
        # Set random weights
        self.links = dict()
        for child in self.children:
            child.add_parent(self)
            link = dict()
            link['child'] = child
            link['weight'] = random.random()
            link['count'] = 0
            self.links[child.name] = link
        self.normalise_weights()
        self.parents = []
        self.type = NodeType.SUM
        self.value = 0.0

    def normalise_weights(self):
        total = np.sum(list(map(lambda l: l['weight'], self.links.values())))
        for link in self.links.values():
            link['weight'] /= total

    def get_value(self, max_mode=False):
        if not max_mode:
            self.value = 0.0
            for child in self.children:
                self.value += self.links[child.name]['weight'] * child.get_value()
            return self.value
        else:
            max_child = {'node': None, 'value': None}
            for child in self.children:
                value = self.links[child.name]['weight'] * child.get_value(max_mode=True)
                if not max_child['value'] or max_child['value'] < value:
                    max_child['node'] = child
                    max_child['value'] = value
            self.value = max_child['value']
            return self.value

    def update_map_weight_counts(self):
        maximum = {'value': 0, 'node': None}
        for child in self.children:
            val = child.value * self.links[child.name]['weight']
            if val >= maximum['value']:
                maximum['value'] = val
                maximum['node'] = child
        self.links[maximum['node'].name]['count'] += 1
        maximum['node'].update_map_weight_counts()

    def normalise_counts_as_weights(self):
        """Normalise the counts by normalising so they sum to one, and then set that as the weights"""
        # Calculates the total by summing the counts of all links
        total = np.sum(list(map(lambda l: l['count'], self.links.values())))
        if total == 0:  # Set all weights and counts to zero
            for link in self.links.values():
                link['weight'] = 0.0
                link['count'] = 0
        else:
            for link in self.links.values():
                link['weight'] = 1.0 * link['count'] / total
                link['count'] = 0


class ProdNode(Node):
    """An SPN product node"""
    def __init__(self, name, children=[]):
        self.name = name
        self.parents = []
        self.links = dict()
        self.children = children
        for child in self.children:
            child.add_parent(self)
            self.links[child.name] = {'child': child, 'count': 0.0}
        self.value = 0.0
        self.type = NodeType.PRODUCT

    def get_value(self, max_mode=False):
        self.value = 1.0
        for child in self.children:
            self.value = self.value * child.get_value(max_mode=max_mode)
        return self.value

    def update_map_weight_counts(self):
        for child in self.children:
            child.update_map_weight_counts()


class LeafNode(Node):
    """An SPN leaf node"""
    def __init__(self, name, value=1.0):
        self.name = name
        self.parents = []
        self.value = value
        self.children = []
        self.type = NodeType.LEAF

    def get_value(self, max_mode=False):
        return self.value

    def update_map_weight_counts(self):
        pass
