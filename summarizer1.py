from sklearn.feature_extraction.text import TfidfVectorizer #Convert a collection of text documents to a matrix of token counts (count vectorizer)
# Convert a collection of raw documents to a matrix of TF-IDF features. (tfidvectorizer)
#Equivalent to CountVectorizer followed by TfidfTransformer.
#TFIDF: TF-IDF (term frequency-inverse document frequency) is a statistical measure that evaluates how relevant a word is to a document in a collection of documents
from spacy.lang.en import English #
import numpy as np #numpy: NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices

nlp = English()
nlp.add_pipe('sentencizer') #sentencizer: Pipeline component for rule-based sentence boundary detection. decides where a sentence begns or ends


def summarizer(text, tokenizer=nlp , max_sent_in_summary=3):

    doc = nlp(text.replace("\n", "")) #Removes all line breaks from a long string of text ie converts all lines into a single line .
    sentences = [sent.text.strip() for sent in doc.sents] # sent.text.strip() : The strip() method removes any leading (spaces at the beginning) and trailing (spaces at the end) characters . 
#Breaks up document by sentences with Spacy. "The dog ran. The cat jumped" into ["The dog ran", "The cat jumped"]. ie gives sentences in a list.
    
    sentence_organizer = {k:v for v,k in enumerate(sentences)}
# sentence_organizer = {k:v for v,k in enumerate(sentences)}: This is a dictionary. A dictionary comprises of two things : Keys and Values.
# Enumerate() method adds a counter to an iterable and returns it in a form of enumerating object. 
# Eg: l1 = ["eat", "sleep", "repeat"] then enumerate(l1)=[(0, 'eat'), (1, 'sleep'), (2, 'repeat')].
    
    vectorizer = TfidfVectorizer(min_df=2,  max_features=None, 
                                        strip_accents='unicode', 
# strip_accents='unicode': Remove accents and perform other character normalization during the preprocessing step. 
# String normalization is removing non-English (ASCII) characters, accents and diacritic marks. Unicode is like python mei ASCII code.
                                        analyzer='word',
#analyzer{‘word’} : Whether the feature should be made of word n-gram or character n-grams. ie wether the sentence should be futher analyzed as n sentences or n characters. 
# If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.
                                        token_pattern=r'\w{1,}',
#token_pattern=r'\w{1,}': Regular expression denoting what constitutes a “token”, only used if analyzer == 'word'.
                                        ngram_range=(1, 3), 
#ngram_range=(1, 3): ngram_rangetuple (min_n, max_n), default=(1, 1)-------
# The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.Only applies if analyzer is not callable.
                                        use_idf=1,smooth_idf=1,
#  use_idf=1,smooth_idf=1: IDF: Inverse Document Frequency, which measures how important a term is.
#Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.
#The reason we need IDF is to help correct for words like “of”, “as”, “the”, etc. since they appear frequently in an English corpus. 
# Thus by taking inverse document frequency, we can minimize the weighting of frequent terms while making infrequent terms have a higher impact.
                                        sublinear_tf=1,
#sublinear_tf=1: sublinear_tf=bool, default=False
#Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf). Sublinear tf-scaling is modification of term frequency.
                                        stop_words = 'english')
# stop_words = 'english': If a list, that list is assumed to contain stop words, 
# all of which will be removed from the resulting tokens. Only applies if analyzer == 'word'.
    
    vectorizer.fit(sentences)
    vectors = vectorizer.transform(sentences)
    scores = np.array(vectors.sum(axis=1)).ravel()
    N = max_sent_in_summary
    top_n_sentences = [sentences[ind] for ind in np.argsort(scores, axis=0)[[::-1]:N]]
    
    top_n = [(sentence,sentence_organizer[sentence]) for sentence in top_n_sentences]
   
    top_n = sorted(top_n, key = lambda x: x[1])
    ordered_scored_sentences = [element[0] for element in top_n]
   
    summary = " ".join(ordered_scored_sentences)
    return summary

