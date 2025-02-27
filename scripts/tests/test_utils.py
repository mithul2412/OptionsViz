import unittest
import numpy as np
from utils import chain_utils

class TestUtils(unittest.TestCase):

    def test_add(self):
        res = chain_utils.add(1,2)
        self.assertAlmostEqual(res, 3, delta=1e-08, msg="Probabilities do not egaul the negative log of the base probability")

if __name__ == '__main__':
    unittest.main()