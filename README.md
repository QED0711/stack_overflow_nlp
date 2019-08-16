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

> If you would like to see the final model (logistic regression, 81% accuracy) in action, see our [companion web app](https://tag-predictor.netlify.com/) for this project.

> _For slide deck summary, see [here](#)_
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