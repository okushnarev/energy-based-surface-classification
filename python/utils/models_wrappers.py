import joblib
from catboost import CatBoostClassifier
import torch

from python.utils.neural_network import NN


class MlModel:
    def __init__(self):
        self.model = None

    def predict(self, data):
        return self.model.predict(data)


class SklearnModel(MlModel):
    def __init__(self, path_to_model):
        """

        :param path_to_model: .joblib file
        """
        super().__init__()
        self.model = joblib.load(path_to_model)


class CatBoostModel(MlModel):
    def __init__(self, path_to_model):
        """

        :param path_to_model: .cbm file
        """
        super().__init__()
        self.model = CatBoostClassifier()
        self.model.load_model(path_to_model)


class NNModel(MlModel):
    def __init__(self, path_to_model, path_to_scaler, input_dim=4):
        """

        :param path_to_model: .pth file
        :param path_to_scaler: .pkl file
        :param input_dim:
        """
        super().__init__()

        self.model = NN(input_dim)
        self.model.load_state_dict(torch.load(path_to_model))

        self.scaler = joblib.load(path_to_scaler)

    def predict(self, data):
        X_test_torch = self.scaler.transform(data)
        X_test_torch = torch.tensor(X_test_torch, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            nn_pred = self.model(X_test_torch)

        return (nn_pred >= 0.5).int()
