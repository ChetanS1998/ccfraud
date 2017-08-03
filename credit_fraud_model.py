# Model builder and runner
from sklearn import linear_model
import numpy as np
class CreditFraudModel(object):

    def __init__(self, weights):
        self._weights = weights

    @property
    def weights(self):
        return self._weights

    def predict(self, observation, model):
        '''
        Given a single observation, this method will return a prediction
        for the target column in the data set
        :param observation: A single row in a numpy ndarray
        :return: Prediction, 1 or 0
        '''
        return model.predict(observation)[0]

class CreditFraudModelBuilder(object):

    def build(self):

        return linear_model.LogisticRegression(C=1e6)

    def train(self, model, training_set, features, target):

        feature_data = training_set[:, features]
        target_data  = training_set[:, target][:,0]
        model.fit(feature_data, target_data)



if __name__=='__main__':
    main()
