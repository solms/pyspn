from NodeType import NodeType


class SPN:
    """The structure for an SPN"""
    def __init__(self, nodes):
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

