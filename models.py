# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, data, embeddings, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.labels = [ex.label for ex in data]
        self.texts = [ex.words for ex in data]
        self.embeddings = embeddings
        self.model = self.DAN(embeddings=embeddings, input_dim=input_dim,
                              hidden_dims=hidden_dims, output_dim=output_dim)
        self.dataset = self.SentimentDataset(texts=self.texts, labels=self.labels,
                                             embedding_fn=self.embeddings.get_embedding)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        outputs = self.model(ex_words)
        _, prediction = torch.max(outputs, dim=0) # Gives index of higher output
        return prediction

    class DAN(nn.Module):
        def __init__(self, embeddings, input_dim, hidden_dims, output_dim):
            super().__init__()
            self.embeddings = embeddings
            self.softmax = nn.Softmax(dim=0)

            # Fully connected layers
            self.fc_layers = nn.Sequential()
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.fc_layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            self.fc_layers.append(nn.Linear(prev_dim, output_dim))

            # Activation function
            self.activation = nn.ReLU()

        def forward(self, x):
            # Fully connected layers with ReLU
            for fc_layer in self.fc_layers:
                x = self.activation(fc_layer(x))

            x = self.softmax(x)

            return x

    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, embedding_fn):
            self.texts = texts
            self.labels = labels
            self.embedding_fn = embedding_fn

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            embedded_words = [self.embedding_fn(word) for word in text]
            avg_embedding = torch.tensor(np.mean(embedded_words, axis=0),
                                         dtype=torch.float32)
            label = self.labels[idx]
            return avg_embedding, label


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
    # Model
    Classifier = NeuralSentimentClassifier(data=train_exs, embeddings=word_embeddings,
                                           input_dim=word_embeddings.get_embedding_length(),
                                           hidden_dims=[32, 64, 128], output_dim=2)
    # Data
    batch_size = 1
    dataloader = DataLoader(Classifier.dataset, batch_size=batch_size, shuffle=True)

    # Set up training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(Classifier.model.parameters(), lr=.00005)
    num_epochs = 20
    running_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        steps = 0
        for texts, labels in dataloader:
            optimizer.zero_grad()
            outputs = Classifier.model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        running_loss.append(epoch_loss/steps)
        print(f"Epoch {epoch} loss: {epoch_loss/steps}")

    return Classifier

