from Node import SumNode, LeafNode, ProdNode
from Layer import Layer
from SPN import SPN

if __name__ == '__main__':
    x1 = LeafNode('x1')
    x1_ = LeafNode('x1_')
    s1 = SumNode('s1', [x1, x1_])
    s2 = SumNode('s2', [x1, x1_])

    x2 = LeafNode('x2')
    x2_ = LeafNode('x2_')
    s3 = SumNode('s3', [x2, x2_])
    s4 = SumNode('s4', [x2, x2_])

    p1 = ProdNode('p1', [s1, s3])
    p2 = ProdNode('p2', [s2, s4])

    s5 = SumNode('s5', [p1, p2])

    spn = SPN([x1, x1_, x2, x2_, s1, s2, s3, s4, s5, p1, p2])

    s1.weights['x1']['value'] = 0.8
    s1.weights['x1_']['value'] = 0.2
    s2.weights['x1']['value'] = 0.6
    s2.weights['x1_']['value'] = 0.4

    s3.weights['x2']['value'] = 0.3
    s3.weights['x2_']['value'] = 0.7
    s4.weights['x2']['value'] = 0.4
    s4.weights['x2_']['value'] = 0.6

    s5.weights['p1']['value'] = 0.35
    s5.weights['p2']['value'] = 0.65

    x1.value = 1.0
    x1_.value = 0.0
    x2.value = 0.0
    x2_.value = 1.0

    print('Root value: ', spn.get_root_value())
