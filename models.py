# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *
from utils import *
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
        self.indexer = embeddings.word_indexer
        self.model = self.DAN(embeddings=embeddings, indexer=embeddings.word_indexer,
                              input_dim=input_dim, hidden_dims=hidden_dims,
                              output_dim=output_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        indices = torch.tensor([self.indexer.index_of(word) for word in ex_words]).to(self.device)
        # print(indices)
        # print(np.shape(indices))
        indices = indices.unsqueeze(0)
        outputs = self.model(indices)
        _, prediction = torch.max(outputs, dim=0) # Gives index of higher output
        return prediction.detach().cpu()

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        batch_size = 10
        dataset = TestDataset(texts=all_ex_words, indexer=self.embeddings.word_indexer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=padding_collate_test)
        all_predictions = []

        self.model.eval()
        for texts in dataloader:
            texts = texts.to(self.device)
            outputs = self.model(texts)
            _, predictions = torch.max(outputs, dim=1)
            all_predictions += predictions.cpu().detach().tolist()

        return all_predictions

    class DAN(nn.Module):
        def __init__(self, embeddings, indexer, input_dim, hidden_dims, output_dim):
            super().__init__()
            self.embeddings = embeddings
            self.embedding_layer = embeddings.get_initialized_embedding_layer(frozen=False)
            self.indexer = indexer
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout()
            self.softmax = nn.Softmax(dim=0)

            # Fully connected layers
            self.fc_layers = nn.Sequential()
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.fc_layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            self.fc_layers.append(nn.Linear(prev_dim, output_dim))

        def forward(self, indices):
            embedded = self.embedding_layer(indices)
            x = torch.mean(embedded, dim=1)

            # Fully connected layers with ReLU
            for fc_layer in self.fc_layers:
                x = self.activation(fc_layer(x))

            x = self.softmax(x)

            return x


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
                                           input_dim=300, hidden_dims=[120, 60],
                                           output_dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Classifier.model.to(device)  # Move model to the GPU if available

    # Data
    batch_size = 7
    dataset = SentimentDataset(data=train_exs, indexer=word_embeddings.word_indexer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=padding_collate)

    # Training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(Classifier.model.parameters(), lr=0.0035)
    num_epochs = 10
    running_loss = []

    for i in range(num_epochs):
        epoch_loss = 0
        steps = 0
        for batch in dataloader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = Classifier.model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # raise Exception("break")

            epoch_loss += loss.item()
            steps += 1

        running_loss.append(epoch_loss/steps)
        print(f"Epoch {i} loss: {epoch_loss/steps}")

    return Classifier

