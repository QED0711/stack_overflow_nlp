import pandas as pd
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ROCAUC

from text_pre_processor import *

import warnings
warnings.filterwarnings("ignore")

class ConstructModel:

    """
    Docstring:

    Generates a pipeline from a series of transformers and models (intended for use with text classification models). 
    Provides a fast and simple means of calculating model performance, generating plots (confusion matrices and AUC/ROC curves)
    and making new predictions. 
    """

    def __init__(self, steps, X, y, fit_on="train"):
        """
            Pipeline initialization and model fitting

            Attributes
            ----------
            steps : list
                A list of transformers and models (like you would input into an sklearn pipeline). 
                Unlike pipelines, you do not need to specify a string name for each transformer/model. 
                Example: [StandardScaler(), PCA(), RandomForestClassifier()]

            X : array of features
                Put in the entire X array of features. A train test split will be performed on initializtion. 

            y : array of targets
                Put in the entire y array of targets. A train test split will be performed on initializtion. 

            fit_on : string (default=train)
                Options are 'train', 'test', and 'all'. Determines what subset of data to fit the model on. 
                Default is 'train'. If 'test', will fit on the test set. If 'all', will fit on the entire dataset.

        """
        steps = [(str(i), steps[i]) for i in range(len(steps))]
        self.pipe = Pipeline(steps)
        self._X = X
        self._y = y
        
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X, self._y, train_size=0.8, random_state=42)

        self._fit(on=fit_on)

        self.tpp = TextPreProcessor()

    def _fit(self, on="train"):

        if on == "train":
            self.pipe.fit(self._X_train, self._y_train)
        elif on == "test":
            self.pipe.fit(self._X_test, self._y_test)
        elif on == "all":
            self.pipe.fit(self._X, self._y)

    def get_score(self):
        """
        Returns
        -------
            calculates and returns the train and test scores of your model
        """
        train_score = self.pipe.score(self._X_train, self._y_train)
        test_score = self.pipe.score(self._X_test, self._y_test)
        print("Train Score:\t", train_score)
        print("Test Score:\t", test_score)

    def get_confusion_matrix(self, on="test"):
        """
        Produces a confusion matrix made through the yellowbrick package.

        Input
        -----
        on : string (default=test)
            Determines which set of data to score and create a confusion matrix on.
            Default is 'test', meaning it will make a confusion matrix of the test results. 
            'train' and 'all' are alternative values. 
        """

        cm = ConfusionMatrix(self.pipe)
        if on =="test": 
            cm.score(self._X_test, self._y_test)
        elif on == "train":
            cm.score(self._X_train, self._y_train)
        elif on == "all":
            cm.score(self._X, self._y)
        
        # graph the confusion matrix with yellowbrick
        cm.poof()

    def get_roc(self, on="test"):
        """
        Produces aAUC/ROC curve graph made through the yellowbrick package

        Input
        -----
        on : string (default=test)
            Determines which set of data to score and create a ROC graph on.
            Default is 'test', meaning it will make a ROC graph of the test results. 
            'train' and 'all' are alternative values. 
        """
        visualizer = ROCAUC(self.pipe)
        if on =="test": 
            visualizer.score(self._X_test, self._y_test)
        elif on == "train":
            visualizer.score(self._X_train, self._y_train)
        elif on == "all": 
            visualizer.score(self._X, self._y)
        
        visualizer.poof()    
        
    def get_prediction(self, text, format="classification"):
        """
        Predicts the classification using the class pipeline created on initialization. 
        Multiple formats are available for the output of the classification. 

        Input
        -----
        text : string (no default, required)
            The string that will be classified

        format : string (default=classification)
            Determines the output format of the method. Default is 'classification', meaning
            it will return a direct classification. Alternative is 'proba', meaning it will
            give the predicted probabilities of each classification option.

        Returns
        -------
        list
            Will return a list containing the single prediction (if format is set to 'classification'), or a list of 
            proabilities associated with each classification (if format is set to 'proba')
        """
        processed_text = self.tpp.process_text(text)
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