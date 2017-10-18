from Node import SumNode, LeafNode, ProdNode
from Layer import Layer
from SPN import SPN
import random


def generate_training_examples(num=1000):
    examples = []
    for i in range(num):
        sample = random.random()
        if sample < 0.2:
            examples.append([0,0,0])
        elif sample < 0.6:
            examples.append([0,1,1])
        elif sample < 0.7:
            examples.append([1,0,1])
        else:
            examples.append([1,1,0])
    return examples


if __name__ == '__main__':
    # x1 = LeafNode('x1')
    # x1_ = LeafNode('x1_')
    # s1 = SumNode('s1', [x1, x1_])
    # s2 = SumNode('s2', [x1, x1_])
    #
    # x2 = LeafNode('x2')
    # x2_ = LeafNode('x2_')
    # s3 = SumNode('s3', [x2, x2_])
    # s4 = SumNode('s4', [x2, x2_])
    #
    # p1 = ProdNode('p1', [s1, s3])
    # p2 = ProdNode('p2', [s2, s4])
    #
    # s5 = SumNode('s5', [p1, p2])
    #
    # spn = SPN([x1, x1_, x2, x2_, s1, s2, s3, s4, s5, p1, p2])

    # x1 = LeafNode('x1')
    # x1_ = LeafNode('x1_')
    # x2 = LeafNode('x2')
    # x2_ = LeafNode('x2_')
    #
    # s1 = SumNode('s1', [x1, x1_])
    # s2 = SumNode('s2', [x1, x1_])
    #
    # p1 = ProdNode('p1', [x2_, s1])
    # p2 = ProdNode('p2', [x2, s2])
    #
    # s3 = SumNode('s3', [p1, p2])
    #
    # spn = SPN([x1_,x1,x2_,x2,s1,s2,s3,p1,p2])

    data = generate_training_examples(1000)
    variables = ['x1','x2','x3']
    spn = SPN()
    spn.create_structure(data, variables)
    # spn.fit(variables, data, epochs=50)
    #
    # x1.value = 1.0
    # x1_.value = 0.0
    # x2.value = 1.0
    # x2_.value = 0.0
    #
    # print(spn.get_root_value())
    #
    # x1.value = 0.0
    # x1_.value = 1.0
    # x2.value = 1.0
    # x2_.value = 0.0
    #
    # print(spn.get_root_value())
