## **Stack Overflow Tag Predictor**
#### Classifying Posts Using NLP 

_See [here](https://tag-predictor.netlify.com/) for the companion web app_
___
#### Authors: 
[Quinn Dizon](https://github.com/QED0711)  
[Mindy Zhou](https://github.com/mzhou356)

___

## Summary

Using raw text data retrieved from _Stack Overflow_ posts, we predict the main programming language tag for each post. 

We begin by performing natural language processing (NLP) using the NLTK library to extract feature data from the raw posts. We then train and measure the accuracy of a number of different machine learning models. 

Our top three models were logistic regression, multinomial NB, and random forest classifier. All produced accuracy scores around 80%. Using all the models together in majority vote, we were able to get about 83% accuracy. 

As a secondary analysis, we attempted to perform topic clustering on the processed dataset. The results for this clustering analysis were inconclusive. 

#### Conclusion

Our final conclusion is that, while we are able to get relatively good results in predicting language, topics within or among languages are numerous, share many common words, and are difficult to distinguish.

> If you would like to see the final model (logistic regression, 81% accuracy) in action, see our [companion web app](https://tag-predictor.netlify.com/) for this project.

> _For a visual slide deck summary, see [here](#)_
___

## Dataset

All data was retrieved directly from Stack Overflow using _Google Query_.

We limited our dataset to a little over 32 thousand unique posts with five of the most popular programming language categories:  

**Java | C# | Javascript | Python| C++**

___

## File Structure

### Final Analysis: 
Our final, high-level analysis can be found in:


> [/notebooks/Stack_Overflow_NLP_Summary_Notebook.ipynb](#)

___
### Cleaned Dataset:

The dataset we used in our final analysis can be found in:
> [/data/final/text_target.pkl](#)
___
### Primary Classes

We wrote custom classes to handle text preprocessing/NLP and the formation and evaluation of our model pipelines. The code for those classes can be found in the respective folders listed below:

- > [/utils/text_pre_processor.py](#)
- > [/utils/construct_model.py](#)

A notebook demonstrating the use of each class can be found in:

> [/notebooks/class_demonstration.ipynb](#)

___

## Acknowledgements 

In doing research for this project, we found the following articles very helpful:

> [Topic Modeling and Latent Dirichlet Allocation (LDA) in Python](https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24)  
> A basic exploration and tutorial for LDA in python

> [Gensim Tutorial – A Complete Beginners Guide](https://www.machinelearningplus.com/nlp/gensim-tutorial/)  
> A guide for text preprocessing/analysis using the Gensim Library