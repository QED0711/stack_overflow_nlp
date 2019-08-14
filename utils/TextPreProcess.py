import re
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')


class TextPreProcessor:
    
    def __init__(self, classifications=["python", "javascript", "java", "c++", "c#"]):
        
        self.classifications = classifications
        self.lemmatizer = WordNetLemmatizer()
      
    
    def text_cleaner(self, post):
        '''
        input:
        post: a string with symbols and punctuations 
        returns:
        cleaned post with all letters to lower, all numbers, white space, and symbols removed
        '''
        pattern = r'[^A-Za-z]+'  # anything that is not letter or space
        processed = re.sub(pattern, ' ', post).strip().lower()
        return processed
    
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return  
        
    def token_lemma(self, post):
        '''
        input:
        post: cleaned post from function text_preprocess
        returns:
        tokenized post with lemmatization with position tags
        stopwords and tags are removed 
        '''
        tokens = word_tokenize(post)
        # stopwords
        stop_words = set(stopwords.words('english'))  # make sure no repeats
        # remove stopwords and remove words that are explicit tags
        words_to_remove = set(self.classifications).union(stop_words)
        # perform pos tag before stop word removal to include more context for pos tags 
        tags = nltk.pos_tag(tokens)
        tags_word_net = [self.get_wordnet_pos(w[1]) for w in tags]
        lem_result = []  # only include nonstop words and target tags 
        for i in range(len(tags_word_net)):
            if tags[i][0] in words_to_remove:  # don't lemmatize unneeded words 
                continue
            if tags_word_net[i]:  # not none 
                lem_result.append(self.lemmatizer.lemmatize(tags[i][0],tags_word_net[i]))
            else:
                lem_result.append(tags[i][0])
        return lem_result
    
    
    def process_text(self, post):
        processed_text = self.text_cleaner(post)
        token_lemma = self.token_lemma(processed_text)
        return " ".join(token_lemma)
    

if __name__ == "__main__":
    # Test to ensure class is working
    tpp = TextPreProcessor()
    tpp.process_text("Hello World, in python")