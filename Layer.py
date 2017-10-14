from NodeType import NodeType


class Layer:
    """The structure for an SPN layer"""
    def __init__(self, nodes):
        self.nodes = nodes

    def add_node(self, node):
        self.nodes.append(node)

    def get_leaves(self):
        return [node for node in self.nodes if node.type == NodeType.LEAF]