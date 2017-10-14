import unittest
from SPN import SPN
from Node import SumNode, LeafNode, ProdNode
from NodeType import NodeType


class TestBasicSPN(unittest.TestCase):

    def setUp(self):
        self.x1 = LeafNode('x1')
        self.x1_ = LeafNode('x1_')
        self.s1 = SumNode('s1', [self.x1, self.x1_])
        self.s2 = SumNode('s2', [self.x1, self.x1_])

        self.assertEqual(self.x1.parents, [self.s1, self.s2])
        self.assertEqual(self.x1_.parents, [self.s1, self.s2])
        self.assertEqual(self.s1.children, [self.x1, self.x1_])
        self.assertEqual(self.s2.children, [self.x1, self.x1_])

        self.x2 = LeafNode('x2')
        self.x2_ = LeafNode('x2_')
        self.s3 = SumNode('s3', [self.x2, self.x2_])
        self.s4 = SumNode('s4', [self.x2, self.x2_])

        self.assertEqual(self.x2.parents, [self.s3, self.s4])
        self.assertEqual(self.x2_.parents, [self.s3, self.s4])
        self.assertEqual(self.s3.children, [self.x2, self.x2_])
        self.assertEqual(self.s4.children, [self.x2, self.x2_])

        self.p1 = ProdNode('p1', [self.s1, self.s3])
        self.p2 = ProdNode('p2', [self.s2, self.s4])

        self.assertEqual(self.s1.parents, [self.p1])
        self.assertEqual(self.s2.parents, [self.p2])
        self.assertEqual(self.s3.parents, [self.p1])
        self.assertEqual(self.s4.parents, [self.p2])
        self.assertEqual(self.p1.children, [self.s1, self.s3])
        self.assertEqual(self.p2.children, [self.s2, self.s4])

        self.s5 = SumNode('s5', [self.p1, self.p2])

        self.assertEqual(self.s5.children, [self.p1, self.p2])
        self.assertEqual(self.p1.parents, [self.s5])
        self.assertEqual(self.p2.parents, [self.s5])

        self.spn = SPN([self.x1, self.x1_, self.x2, self.x2_,
                        self.s1, self.s2, self.s3, self.s4, self.s5, self.p1, self.p2])

    def set_leaf_values(self):
        """Set up the leaf values for the tests"""
        self.s1.links['x1']['weight'] = 0.8
        self.s1.links['x1_']['weight'] = 0.2
        self.s2.links['x1']['weight'] = 0.6
        self.s2.links['x1_']['weight'] = 0.4

        self.s3.links['x2']['weight'] = 0.3
        self.s3.links['x2_']['weight'] = 0.7
        self.s4.links['x2']['weight'] = 0.4
        self.s4.links['x2_']['weight'] = 0.6

        self.s5.links['p1']['weight'] = 0.35
        self.s5.links['p2']['weight'] = 0.65

        self.x1.value = 1.0
        self.x1_.value = 0.0
        self.x2.value = 0.0
        self.x2_.value = 1.0

    def test_calc_root_value(self):
        """Test calculating the value at the root node"""
        self.set_leaf_values()
        self.spn.get_root_value()

        self.assertEqual(self.s1.value, 0.8)
        self.assertEqual(self.s2.value, 0.6)
        self.assertEqual(self.s3.value, 0.7)
        self.assertEqual(self.s4.value, 0.6)
        self.assertEqual(self.p1.value, self.s1.value * self.s3.value)
        self.assertEqual(self.p2.value, self.s2.value * self.s4.value)
        self.assertEqual(
            self.s5.value,
            self.s5.links[self.p1.name]['weight'] * self.p1.value +
            self.s5.links[self.p2.name]['weight'] * self.p2.value
            )
        self.assertEqual(self.spn.get_root_value(), self.s5.value)

    def test_calc_max_route(self):
        """Test calculating the route through the SPN when changing sum nodes to max nodes"""
        self.set_leaf_values()
        self.spn.get_root_value(max_mode=True)  # Equivalent to bottom-up max pass
        self.spn.calculate_map_route_counts()

        self.assertEqual(self.s1.value, 0.8)
        self.assertEqual(self.s2.value, 0.6)
        self.assertEqual(self.s3.value, 0.7)
        self.assertEqual(self.s4.value, 0.6)

        self.assertAlmostEqual(self.p1.value, 0.56)
        self.assertAlmostEqual(self.p2.value, 0.36)

        self.assertAlmostEqual(self.s5.value, 0.234)

        self.assertEqual(self.s1.links['x1']['count'], 0.0)
        self.assertEqual(self.s1.links['x1_']['count'], 0.0)
        self.assertEqual(self.s2.links['x1']['count'], 1.0)
        self.assertEqual(self.s2.links['x1_']['count'], 0.0)

        self.assertEqual(self.s3.links['x2']['count'], 0.0)
        self.assertEqual(self.s3.links['x2_']['count'], 0.0)
        self.assertEqual(self.s4.links['x2']['count'], 0.0)
        self.assertEqual(self.s4.links['x2_']['count'], 1.0)

        self.assertEqual(self.s5.links['p1']['count'], 0.0)
        self.assertEqual(self.s5.links['p2']['count'], 1.0)

        self.spn.normalise_counts_as_weights()

        # Expect all counts to be zero after normalisation
        for node in self.spn.nodes:
            if node.type == NodeType.SUM:
                for link in node.links.values():
                    self.assertEqual(link['count'], 0.0)
