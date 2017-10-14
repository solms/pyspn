from enum import Enum


class NodeType(Enum):
    """Different types of SPN nodes"""
    SUM = 'SUM'
    PRODUCT = 'PRODUCT'
    LEAF = 'LEAF'
