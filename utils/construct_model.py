import pandas as pd
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ROCAUC

import warnings
warnings.filterwarnings("ignore")

class ConstructModel:

    def __init__(self, steps, X, y, fit_on="train"):

        steps = [(str(i), steps[i]) for i in range(len(steps))]
        self.pipe = Pipeline(steps)
        self.X = X
        self.y = y
        
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=42)

        self._fit(on=fit_on)

    def _fit(self, on="train"):
        if on == "train":
            self.pipe.fit(self._X_train, self._y_train)
        elif on == "test":
            self.pipe.fit(self._X_test, self._y_test)
        elif on == "all":
            self.pipe.fit(self.X, self.y)

    def get_score(self):
        train_score = self.pipe.score(self._X_train, self._y_train)
        test_score = self.pipe.score(self._X_test, self._y_test)
        print("Train Score:\t", train_score)
        print("Test Score:\t", test_score)

    def get_confusion_matrix(self, on="test"):
        cm = ConfusionMatrix(self.pipe)
        if on =="test": 
            cm.score(self._X_test, self._y_test)
        elif on == "train":
            cm.score(self._X_train, self._y_train)
        elif on == "all":
            cm.score(self.X, self.y)
        
        # graph the confusion matrix with yellowbrick
        cm.poof()

    def get_roc(self, on="test"):
        
        visualizer = ROCAUC(self.pipe)
        if on =="test": 
            visualizer.score(self._X_test, self._y_test)
        elif on == "train":
            visualizer.score(self._X_train, self._y_train)
        elif on == "all": 
            visualizer.score(self.X, self.y)
        
        visualizer.poof()    
        
    def get_prediction(self, processed_text, format="classification"):
        if format == "classification":
            return self.pipe.predict([processed_text])
        elif format == "proba":
            classes = self.pipe.classes_
            return list(zip(classes, list(self.pipe.predict_proba([processed_text])[0])))
    
    
if __name__ == "__main__":
    import pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer

    with open('../data/interim/text_target.pkl', 'rb') as f:
        text_target = pickle.load(f)

    X = text_target.cleaned_text
    y = text_target.target

    vec = TfidfVectorizer()
    lr = LogisticRegression(random_state=42)

    lr_pipe = ConstructModel([vec, lr], X, y, fit_on="train")   

    print(lr_pipe.get_score())