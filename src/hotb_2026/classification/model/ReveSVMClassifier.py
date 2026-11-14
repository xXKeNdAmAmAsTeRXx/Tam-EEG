from typing import List

from transformers import AutoModel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


class ReveSVMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 electrodes: List[str],
                 pos_embedding_model_name: str = "brain-bzh/reve-positions",
                 signal_embedding_model_name: str = "brain-bzh/reve-base",
                 C:float = 1.0,
                 kernel:str = "rbf",
                 gamma:float | str = "scale",
                 ):

        # SVM Settings
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

        # Encode Electrodes
        pos_embedding_model = AutoModel.from_pretrained(pos_embedding_model_name)
        self.electrodes_pos = pos_embedding_model(electrodes)

        # Signal Encoding Model
        self.signal_embedding_model = AutoModel.from_pretrained(signal_embedding_model_name, trust_remote_code=True)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass