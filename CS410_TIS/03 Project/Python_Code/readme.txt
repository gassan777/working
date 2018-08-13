Overview

This project is a demonstration of vectorizing text with Google’s word2vec algorithm implemented in the gensim module for Python. This is an algorithm for building vector representations of words (i.e. word embeddings). The vectors describe semantically how word are used in context – words used in text in a similar way have similar vector representations. You can use words mapped into vector space to look for words having similar semantics. The model takes a list of sentences which, in turn, must be lists of words.

For reference, the API documentation for the wrod2vec implementation in Python is provided here: https://radimrehurek.com/gensim/models/word2vec.html. Some word2vec functions are deprecated as stated in this API doc, and you can find alternatives in the KeyedVectors submodule here: https://radimrehurek.com/gensim/models/keyedvectors.html. A simple, but very useful tutorial by the author of the module can be found here: https://rare-technologies.com/word2vec-tutorial/
 
Installation

For this project I tested and used different Python modules dealing with NLP tasks. They include:
- gensim
- cython (speeds up gensim a lot!)
- NLTK
- pattern

Install gensim:
$ sudo pip install --upgrade gensim
More detailed information:
https://radimrehurek.com/gensim/install.html

Gensim depends on Python >= 2.6, NumPy >= 1.3, SciPy >= 0.7
Python is usually part of the Linux distribution. If you don’t have the last two, install as follows on Debian/Ubuntu:
$ sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
More detailed information and other OS:
https://www.scipy.org/install.html

Install Cython:
$ sudo pip install Cython
Unlike most Python software, Cython requires a C compiler to be present in the system. The details of getting a C compiler vary according to the system and can be found here:
http://cython.readthedocs.io/en/latest/src/quickstart/install.html

Install pattern:
$ sudo pip install pattern
More detailed information:
https://github.com/clips/pattern

Install NLTK and NLTK corpus:
$ sudo pip install -U nltk
$ python -m nltk.downloader all

If you use anaconda, you can run these commands:
$ conda install -c anaconda gensim
$ conda install -c anaconda cython
$ conda install -c anaconda numpy
$ conda install -c anaconda nltk
$ conda install -c anaconda scipy

Process Description

As input my project takes several general (raw) text files in different categories. For simplicity, I selected two categories: Facilities and Geoscience. But you can add more folders with raw text files from different categories, add empty folders with category names in the “models” directory, and add categories and corresponding paths to the code. This is relatively easy and will enable you to have more than just two categories.

The CreateVocab() function preprocesses every raw text file one by one in each category using some basic operations, e.g. tokenizing, removing hyphens from pairs of hyphenated words, which I believe is useful for semantic analysis, and normalizing words by using the normalize() function which, in turn, converts words to lower case and singular form, removes all non-letters and stop words, and conducts stemming. The preprocessed text is saved as the same lines in the same number of files (with modified file names). As an option, by introducing small changes to the code, it can be saved as one line in the same number of files or one line in just one file representing the entire topic/category.

I used the Snowball stemmer and stop words removal from NLTK. I also used pattern.en to tokenize and singularize text. I found the Snowball stemmer not very efficient for the small scope of my project. As for the pattern’s singularizer, it does a good job, but requires a substantial list of exceptions, e.g. it reduces gas to ga thinking gas is plural. This functionality is provided in the module, but I need to create a list of exceptions on my own. Also, the singulizer probably needs to be coupled with a POS tagger (not part of the project) because it tends to remove the final “s” in adjectives and other parts of speech. Just removing the final “s” is a pretty simplistic approach.

After this, the ModelTrain() function initializes and trains word2vec models using the preprocessed text in the object of the MySentences class which streams data for the models. The models for each category are then saved in the “models” folder, in corresponding subfolders. I used the MySentences class, which is a memory-friendly iterator, from this word2vec tutorial: https://rare-technologies.com/word2vec-tutorial/

Once this is done, the code from the remaining portion of __main__ starts executing. It introduces two sample texts from the two categories. Then, it iteratively loads each of the two models with the in-built word2vec load() function, normalizes each sample text using the same normalize() function that was used for creating by CreateVocab() in order to compare apples with apples, and then calculates the log-likelihood of each sample text given each model using the in-built word2vec score() function. Currently, score is implemented only for the hierarchical softmax scheme, so word2vec needs to run with hs=1 and negative=0. The resulting four log-likelihoods are printed in the end along with associated information.

This is a general overview. In addition to this document, the code itself is heavily commented describing every meaningful operation, so I hope you can clearly understand every step I made. Once you preprocess raw text files and train models, there is no need to run the entire program each time you want to evaluate the log-likelihoods for these or other sample phrases. At that point you can simply comment CreateVocab() and ModelTrain() (put the comment sign # before them) in __main__, eliminating these two stages, and use the models saved to disk by loading them using the remaining code in __main__.

NOTES
1) Logging of computer actions is currently turned on in line 14. It is meant for error analysis and more detailed description of the process. You can comment it to turn it off.

2) CreateVocab() preprocesses raw text files. If you try it on different raw text files, there may be a warning (which I used to get in PyCharm) about being unable to decode certain Unicode characters due to the way Python 2.7 handles Unicode, but they did not interrupt my program or impact the results. This problem is resolved in Python 3. It is described here: https://wiki.python.org/moin/UnicodeDecodeError

4) One of the things that still needs to be done is a better control of the folders where files are read and saved. I ran out of time and did not do this, so now each time you want to create the vocabulary and train models by running CreateVocab() and ModelTrain(), you have to delete the ‘txt’ folder, create a copy the ‘txt_backup’ folder with the original raw text files, and rename it to “txt”.

5) Also, I did not have enough time to fully interpret the log-likelihood results for comparison purposes. But at least, I understand that they are consistent meaning that when a model in a certain category evaluates the sample text from the same category, the results are in the same range for both categories and vice versa.
