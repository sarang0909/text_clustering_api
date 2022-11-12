"""A Predictor module to load model and get prediction from tf-idf custom ml model.

"""
import pickle
from sentence_transformers import SentenceTransformer
from src.inference.predictor import Predictor


class SklearnKmeansPredictor(Predictor):
    """A child class to load model and get output

    Args:
        Predictor (Predictor): Parent class

    """

    model = None

    def __init__(self) -> None:

        self.tokenizer = SentenceTransformer("all-MiniLM-L6-v2")

    def get_model(self):
        """A method to load model

        Returns:
            model: trained model
        """

        if self.model is None:
            with open(
                "src/models/sklearn_kmeans_model.pkl", "rb"
            ) as model_file:
                self.model = pickle.load(model_file)
        return self.model

    def get_model_output(self, input_data):
        """A method to get model output from given text input
        Args:
            input_data (text):input text data

        Returns:
            output: model prediction
        """

        input_vector = self.tokenizer.encode(input_data)
        output = self.get_model().predict(input_vector.reshape(1, -1))[0]
        return output
