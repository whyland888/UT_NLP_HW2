# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """
    def __init__(self, data, embeddings):
        super().__init__()
        self.labels = [ex.label for ex in data]
        self.texts = [ex.words for ex in data]
        self.embeddings = embeddings
        self.model = self.DAN(embeddings)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        outputs = self.model(ex_words)
        prediction = torch.max(outputs, 1)[1]  # Gives index of higher output
        return prediction

    class DAN(nn.Module):
        def __init__(self, embeddings):
            super().__init__()
            # Network layers
            self.embeddings = embeddings
            self.embedding_layer = embeddings.get_initialized_embedding_layer()
            self.embedding_length = embeddings.get_embedding_length()
            self.fc1 = nn.Linear(self.embedding_length, 32)
            self.fc2 = nn.Linear(32, 2)

        def forward(self, text):
            embedded_words = [self.embeddings.get_embedding(word) for word in text]
            avg_embedding = torch.tensor(np.mean(embedded_words, axis=0),
                                         dtype=torch.float32)
            fc1_out = self.fc1(avg_embedding)
            output = self.fc2(fc1_out)
            return output


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """
    Classifier = NeuralSentimentClassifier(data=train_exs, embeddings=word_embeddings)
    n_iters = 20

    for i in range(n_iters):


    # Create training loop

    raise NotImplementedError

