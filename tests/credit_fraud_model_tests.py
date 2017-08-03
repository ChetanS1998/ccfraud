import unittest
from credit_fraud_model import CreditFraudModelBuilder
import numpy as np

class ModelTests(unittest.TestCase):

    def generate_sample(self):
        observation_1 = [10, 20, 30, 40, 50, 10]
        observation_2 = [10, 20, 30, 40, 50, 10]
        observation_3 = [10, 20, 30, 40, 50, 10]
        observation_4 = [10, 20, 30, 40, 50, 10]
        observation_5 = [10, 20, 30, 40, 50, 10]
        observation_6 = [1, 2, 3, 4, 5, 1]
        sample = np.array([
            observation_1,
            observation_2,
            observation_3,
            observation_4,
            observation_5,
            observation_6
        ])
        return sample

    def test_simple_classification(self):
        sample = self.generate_sample()
        model = CreditFraudModelBuilder().build()
        CreditFraudModelBuilder().train(model, sample, [0, 1, 2, 3, 4], [5])
        observation_x = [10, 15, 20, 25, 30]
        prediction = model.predict(observation_x)
        self.assertEqual(10, prediction)

    def test_outlier_classification(self):
        sample = self.generate_sample()
        model = CreditFraudModelBuilder().build()
        CreditFraudModelBuilder().train(model, sample, [0, 1, 2, 3, 4], [5])
        observation_x = [1, 2, 3, 4, -1]
        prediction = model.predict(observation_x)
        self.assertEqual(1, prediction)

if __name__=='__main__':
    unittest.main()