import gensim
import logging
import multiprocessing
import os
import re
import sys
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from pattern.en import tokenize, singularize
from time import time

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)     #logging for error analysis
#nltk.download('stopwords')                             #this was used only the first time; the import command was not enough

def normalize(sentence):                                #normalize sentence; called from CreatVocab and __main__ (to compare apples with apples)
    mysentence = []
    for word in sentence:
        word = word.lower()                             #make every word lower case
        word = filter(lambda c: c.isalpha(), word)      #remove non-chars
        word = word.replace('edit', '')                 #cleaning annoying wikipedia hyperlinks
        word = word.replace('pdf', '')                  #same
        word = singularize(word, custom={'gas':'gas'})  #convert to singular, do not change gas to ga
        word = SnowballStemmer("english").stem(word)    #stemming
        stopWords = set(stopwords.words('english'))     #to be used below to remove stopwords
        if (len(word) == 0) or (len(word) == 1):        #delete empty or single-char words (useless for semantic analysis)
            #print('deleting' + ' ' + word)             #checkpoint for error analysis (hereinafter, checkpoint)
            del word
        elif word in stopWords:                         #remove stopwords
            del word
        else:
            #print(word)                                #checkpoint
            word = word.decode('utf-8')                 #word2vec accepts unicode strings
            mysentence.append(word)                     #append to returned sentence (list of str)
    #print(mysentence)                                  #checkpoint
    return mysentence

def CreateVocab():                                          #Creating vocabulary from multiple original text files
    start = time()                                          #time control
    path = 'txt/'                                           #path for multiple original text files
    for root, dirs, files in os.walk(path):
        for filename in files:
            file1 = root + '/' + filename                   #file to read data from
            file2 = root + '/' + 'prep' + filename          #one-line file to write prepped data to, facilitates work with multiple files
            raw = open(file1).read()                        #convert original text file to str
            print('Prepping' + ' ' + file1)                 #checkpoint
            sentences = tokenize(raw)                       #tokenize original file (list of sentences, each sentence = str)
            text = []                                       #to be used as resulting normalized "list of lists of str" (list of sentences, each sentence = list of str words) for the entire file
            for sentence in sentences:
                sentence = re.sub(r'-', ' ', sentence)      #split hyphenated pairs of word (useful for semantic analysis)
                toksent  = re.split(r'\s+', sentence)       #tokenize each sentence, each sentence = list of str (words)
                toksent = normalize(toksent)                #normalize tokenized sentence
                text.append(toksent)                        #add to normalized "list of lists of str"
            with open(file2, 'w+') as f:                    #write vocabulary to one-line file, facilitates work with multiple original files
                for mysent in text:
                    for myword in mysent:
                        f.write(myword + ' ')
                    f.write('\n')
            os.remove(file1)                                #delete non-normalized original file that has been processed
    print                                                   #for better representation of printed results
    finish = time()
    print("Vocabulary created in: %d seconds" % (finish - start))   #time spent to create the vocabulary
    print

class MySentences(object):                                  #class yiedling a stream of tokens for the word2vec model
    def __init__(self, dirname):                            #initialization of an object of the class
        self.dirname = dirname

    def __iter__(self):                                     #memory-friendly iterator, streams tokens (words) from each one-line file in a given category
        for fname in os.listdir(self.dirname):              #credits: Rare Technologies, https://rare-technologies.com/word2vec-tutorial/
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

if __name__ == '__main__':

    CreateVocab()                                               #Creaing vocabulary

    inpath = ['txt/facilities/', 'txt/geoscience/']             #path for processed one-line files with vacabulary (currently two groups of files in separate directories)
    outpath = ['models/facilities/', 'models/geoscience/']      #path for trained models, currently two models in separate directories, each corresponding to a group of one-line files
    category = ['facilities', 'geoscience']                     #two categories for comparison, each category is related to a group of one-line vacab files and a traind model; names are self-explanatory

    for i in range (len(inpath)):
        start = time()
        sentences = MySentences(inpath[i])                      #creating an object of MySentences class
        model = gensim.models.Word2Vec(sentences,               #training a model
                                        size=10,                #the dimensionality of the feature vectors
                                       window=5,                #the maximum distance between the current and predicted word within a sentence
                                     min_count=2,               #ignore words with total frequency lower than this
                                     workers=multiprocessing.cpu_count(),   #using the max number of processing cores on your machine
                                     hs = 1,                    #Only score for hierarchical softmax scheme is implemented,
                                     negative = 0)              #so word2vec needs to run with hs=1 and negative=0.
                                                                # if hs=0 (default), and negative !=0 - negative sampling
        model.save(outpath[i] + category[i])                    #saving the current model to a corresponding directory
        finish = time()
        print("Model for category \"{}\" created and saved in: {} seconds".format(category[i], (finish - start)))   #time spent to create a model

    print
    toclf = []                                                  #list of str to compare with model (establish the probability of the model generating the str)
    toclf.append("When produced, oil and gas are transported by pipelines to a refinery or gas processing plant \
        and then offloaded onto tanker for delivery to end users of petroleum products")                        #sample text representing the facilities category
    toclf.append("A seismic array is usually a group of seismic sources towed by a geophysical vessel \
            as part of exploration activies at an oil and gas field")                                           #sample text representing the geoscience category

    for i in range (len(outpath)):                              #iteration by model
        mymodel = gensim.models.Word2Vec.load(outpath[i]+ category[i])      #loading of trained model from file
        for j in range (len(category)):                         #iteration by category
            myinput = toclf[j].split()                          #convert text for comparison to list of str
            myinput = normalize(myinput)                        #normalize text for comparison
            score = mymodel.score([myinput], total_sentences=1000)                    #score = probability that the text for comparison was generated by the model
            print("Model in \"{}\". Score for text from ground-truth category \"{}\":".format(outpath[i], category[j]))
            print(score)                                        #print modelled probability and ground-truth for each text for comparison