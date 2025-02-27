import unittest
import numpy as np
from utils import chain_utils

class TestUtils(unittest.TestCase):

    def test_add(self):
        res = chain_utils.add_nums(1,2)
        self.assertAlmostEqual(res, 3, delta=1e-08, msg="numbers are not equal")

if __name__ == '__main__':
    unittest.main()