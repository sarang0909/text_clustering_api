"""A factory module to get predictor class based on type"""
from src.inference.nltk_kmeans_predictor import NLTLKMeansPredictor

from src.inference.sklearn_kmeans_predictor import SklearnKmeansPredictor


def get_predictor(model_type):
    """A method to retun Predictor class object

    Args:
        model_type (str): Model type

    Returns:
        Predictor: A predictor class
    """
    if model_type is None:
        return None
    elif model_type == "nltk_kmeans":
        return NLTLKMeansPredictor()
    elif model_type == "sklearn_kmeans":
        return SklearnKmeansPredictor()
