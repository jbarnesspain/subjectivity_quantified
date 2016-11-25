import numpy as np
import codecs
import re
import os
import nltk
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
import sys
sys.path.append('../../BilBowa_Experiments')
from MyMetrics import *



class Backoff_Sentence_Polarity_Dataset(object):
    """This class takes as input the directory that holds the dataset and
    the word vector model that is used to transform this data. Parameters are as
    follows:

    one_hot: the y labels are one hot vectors where the correct class is 1 and
             all others are 0. Default is True.

    dtype: the dtype of the np.array for each word vector. Default is np.float32.

    rep: this determines how the word vectors are represented. The following are options:

         sentence2vec: each sentence is represented by one vector, which is simply
                       the sum of each of the word vectors in the sentence.

         averageWordVecs: each sentence is represented as the average of all of the
                          word vectors in the sentence.

         getWordVecs: each sentence is a list of the word vectors in the sentence. This
                      approach does not add or average them, but padding must be used
                      during training.
    """

    
    def __init__(self, DIR, model, backoff_model, one_hot=True,
                 dtype=np.float32, rep='avg'):

        self.not_found_in_backoff = []
        self.not_found_in_extracted = []
        self.rep = rep

        """
        train_neg = self.getMyData(os.path.join(DIR, 'train/neg.txt'), 0, model, backoff_model)
        train_pos = self.getMyData(os.path.join(DIR, 'train/pos.txt'), 1, model, backoff_model)
        dev_neg = self.getMyData(os.path.join(DIR, 'dev/neg.txt'), 0, model, backoff_model)
        dev_pos = self.getMyData(os.path.join(DIR, 'dev/pos.txt'), 1, model, backoff_model)
        test_neg = self.getMyData(os.path.join(DIR, 'test/neg.txt'), 0, model, backoff_model)
        test_pos = self.getMyData(os.path.join(DIR, 'test/pos.txt'), 1, model, backoff_model)

       

        traindata = train_neg + train_pos
        devdata = dev_neg + dev_pos
        testdata = test_neg + test_pos
        """
        dataneg = self.getMyData(os.path.join(DIR, 'neg.txt'), 0, model, backoff_model)
        datapos = self.getMyData(os.path.join(DIR, 'pos.txt'), 1, model, backoff_model)
        traindata = dataneg[:int((len(dataneg) * .75))] + datapos[:int((len(datapos) * .75))]
        devdata = dataneg[int((len(dataneg) * .75)):int((len(dataneg) * .85))] + datapos[int((len(datapos) * .75)):int((len(dataneg) * .85))]
        testdata = dataneg[int((len(dataneg) * .85)):]+ datapos[int((len(datapos) * .85)):]
        
        Xtrain = [data for data, y in traindata]
        if one_hot == True:
            ytrain = [self.to_array(y,2) for data,y in traindata]
        else:
            ytrain = [y for data, y in traindata]
        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)

        Xdev = [data for data, y in devdata]
        if one_hot == True:
            ydev = [self.to_array(y,2) for data,y in devdata]
        else:
            ydev = [y for data, y in devdata]
        Xdev = np.array(Xdev)
        ydev = np.array(ydev)

        Xtest = [data for data, y in testdata]
        if one_hot == True:
            ytest = [self.to_array(y, 2) for data,y in testdata]
        else:
            ytest = [y for data,y in testdata]

        Xtest = np.array(Xtest)
        ytest = np.array(ytest)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels"""
        integer = integer -1
        return np.array(np.eye(num_labels)[integer])

    def finish_tokenizing(self, tokens):
        semi_tokens = ' '.join(tokens)
        re_tokenized = re.sub("'", " ' ", semi_tokens)
        re_tokenized = re.sub('-', ' ', re_tokenized)
        re_tokenized = re.sub('/', ' ', re_tokenized)
        return re_tokenized.split()

    def averageWordVecs(self, sentence,model, backoff_model=None):
        """Returns a vector which is the
        averate of all of the vectors in
        the sentence"""
        sent = np.array(np.zeros((model.vector_size)))
        words = self.finish_tokenizing(nltk.word_tokenize(sentence))
        sent_length = len(words)
        
        for w in words:
            try:
                sent += model[w]
            except KeyError:
                self.not_found_in_extracted.append(w)
                try:
                    sent += backoff_model[w]
                except KeyError:
                    self.not_found_in_backoff.append(w)
                    sent_length -= 1
                    pass
            except TypeError:
                self.not_found_in_extracted.append(w)
                try:
                    sent += backoff_model[w]
                except KeyError:
                    self.not_found_in_backoff.append(w)
                    sent_length -= 1
                    pass
                
        return sent/sent_length

    def getMyData(self, fname, label, model, backoff_model):
        if self.rep == 'avg':
            representation = self.averageWordVecs
        data= []
        for sent in codecs.open(fname, 'r', encoding='utf8', errors='replace'):
            data.append((representation(sent,model,backoff_model), label))
        return data

def test_oov_words(oov_words):
    pos_tagged = nltk.pos_tag(oov_words, 'universal')
    adjs = [x for x,t in pos_tagged if t == 'ADJ']
    adv = [x for x,t in pos_tagged if t == 'ADV']
    nns = [x for x,t in pos_tagged if t =='NOUN']
    verbs = [x for x,t in pos_tagged if t =='VERB']
    percent_noun = (len(nns)/len(pos_tagged)) * 100
    percent_adj = (len(adjs)/len(pos_tagged)) * 100
    percent_verb = (len(verbs)/len(pos_tagged))* 100
    percent_adv = (len(adv)/len(pos_tagged))* 100
    total = len(pos_tagged)
    return (percent_noun, percent_adj, percent_verb,
            percent_adv, total)

def create_classifier():
    clf = Sequential()
    clf.add(Dense(input_dim=300, output_dim=800, activation='relu'))
    clf.add(Dropout(.3))
    clf.add(Dense(input_dim=800, output_dim=800, activation='relu'))
    clf.add(Dropout(.3))
    clf.add(Dense(input_dim=800, output_dim=800, activation='relu'))
    clf.add(Dropout(.3))
    clf.add(Dense(input_dim=800, output_dim=2, activation='softmax'))
    clf.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return clf

def create_svm():
    clf = LinearSVC()
    return clf

def test_dataset(dataset, classifier):
    history = classifier.fit(dataset._Xtrain, dataset._ytrain, validation_data=[dataset._Xdev, dataset._ydev],
                      nb_epoch=10, verbose=0, batch_size=32)
    pred = classifier.predict(dataset._Xtest)
    mm = MyMetrics(dataset._ytest, pred, label_names=['N', 'P'], labels=[0,1])
    mm.print_metrics()

def test_svm_dataset(dataset, classifier):
    classifier.fit(dataset._Xtrain, dataset._ytrain)
    pred = classifier.predict(dataset._Xtest)
    acc = accuracy_score(dataset._ytest, pred)
    prec = precision_score(dataset._ytest, pred)
    rec =recall_score(dataset._ytest, pred)
    f1 = f1_score(dataset._ytest, pred)
    return acc, prec, rec, f1

def test_models(lang_models):

    os.chdir('/home/jbarnes/Documents/Experiments/Subjectivity_Quantified/')
    backoff = Word2Vec.load('./LanguageModels/wikipedia/model1')

    results = []
    
    for lmd in lang_models:
        print('Training classifier on %s' % lmd)
        print()
        lang_model = Word2Vec.load('./LanguageModels/'+lmd+'/model1')
        dataset = Backoff_Sentence_Polarity_Dataset('/home/jbarnes/Downloads/rt-polaritydata/rt-polaritydata/', lang_model, backoff)
        classifier = create_classifier()
        test_dataset(dataset, classifier)
        
        

    return results

if __name__ == "__main__":
    
    models = ['extracted_subjectivity_corpora/multiun/high_precision/',
              'extracted_subjectivity_corpora/multiun/high_recall',
              'extracted_subjectivity_corpora/europarl/high_precision',
              'extracted_subjectivity_corpora/europarl/high_recall',
              'extracted_subjectivity_corpora/wikipedia/high_precision/',
              'extracted_subjectivity_corpora/wikipedia/high_recall',
              'wikipedia']
    test_models(models)
