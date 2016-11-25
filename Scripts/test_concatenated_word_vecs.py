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
from test_word_vecs import Sentence_Polarity_Dataset
import sys
sys.path.append('../../BilBowa_Experiments')
from MyMetrics import *
from BilBowa_reader import WordVecs

class Concat_Sentence_Polarity_Dataset(object):
    
    def __init__(self, DIR, model, model2, one_hot=True,
                 dtype=np.float32, rep='avg'):

        self.rep = rep
        self.one_hot = one_hot

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model)
        Xtrain2, Xdev2, Xtest2, ytrain2, ydev2, ytest2 = self.open_data(DIR, model2)

        self.CONCX_train = self.concatenate(Xtrain, Xtrain2)
        self.CONCX_dev = self.concatenate(Xdev, Xdev2)
        self.CONCX_test = self.concatenate(Xtest, Xtest2)
    

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._Xtrain2 = Xtrain2
        self._ytrain2 = ytrain2
        self._Xdev2 = Xdev2
        self._ydev2 = ydev2
        self._Xtest2 = Xtest2
        self._ytest2 = ytest2
        self._num_examples = len(self._Xtrain)
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def concatenate(self, x1, x2):
        return np.concatenate((x1, x2), axis=1)

    def open_data(self, DIR, model):
        dataneg = self.getMyData(os.path.join(DIR, 'neg.txt'), 0, model)
        datapos = self.getMyData(os.path.join(DIR, 'pos.txt'), 1, model)
        traindata = dataneg[:int((len(dataneg) * .75))] + datapos[:int((len(datapos) * .75))]
        devdata = dataneg[int((len(dataneg) * .75)):int((len(dataneg) * .85))] + datapos[int((len(datapos) * .75)):int((len(dataneg) * .85))]
        testdata = dataneg[int((len(dataneg) * .85)):]+ datapos[int((len(datapos) * .85)):]
        
        Xtrain = [data for data, y in traindata]
        if self.one_hot == True:
            ytrain = [self.to_array(y,2) for data,y in traindata]
        else:
            ytrain = [y for data, y in traindata]
        Xtrain = np.array(Xtrain)
        ytrain = np.array(ytrain)

        Xdev = [data for data, y in devdata]
        if self.one_hot == True:
            ydev = [self.to_array(y,2) for data,y in devdata]
        else:
            ydev = [y for data, y in devdata]
        Xdev = np.array(Xdev)
        ydev = np.array(ydev)

        Xtest = [data for data, y in testdata]
        if self.one_hot == True:
            ytest = [self.to_array(y, 2) for data,y in testdata]
        else:
            ytest = [y for data,y in testdata]

        Xtest = np.array(Xtest)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

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

    def averageWordVecs(self, sentence,model):
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
                pass
            except TypeError:
                pass
                
        return sent/sent_length

    def getMyData(self, fname, label, model):
        if self.rep == 'avg':
            representation = self.averageWordVecs
        data= []
        for sent in codecs.open(fname, 'r', encoding='utf8', errors='replace'):
            data.append((representation(sent,model), label))
        return data

def create_classifier(in_dim=300):
    clf = Sequential()
    clf.add(Dense(input_dim=in_dim, output_dim=800, activation='relu'))
    clf.add(Dropout(.3))
    clf.add(Dense(input_dim=800, output_dim=800, activation='relu'))
    clf.add(Dropout(.3))
    clf.add(Dense(input_dim=800, output_dim=800, activation='relu'))
    clf.add(Dropout(.3))
    clf.add(Dense(input_dim=800, output_dim=2, activation='softmax'))
    clf.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return clf

def test_dataset(dataset, classifier):
    history = classifier.fit(dataset.CONCX_train, dataset._ytrain, validation_data=[dataset.CONCX_dev, dataset._ydev],
                      nb_epoch=10, verbose=0, batch_size=32)
    pred = classifier.predict(dataset.CONCX_test)
    mm = MyMetrics(dataset._ytest, pred, label_names=['N', 'P'], labels=[0,1],
                   average='macro')
    prec, rec, f1 = mm.get_scores()
    acc = mm.accuracy()
    print('\n\tprecision: {0}\n\trecall: {1}\n\tf1: {2}\n\tacc: {3}\n\n'.format((prec,
                                                                   rec,
                                                                   f1,
                                                                   acc)))

def test_models(lang_models):

    os.chdir('/home/jbarnes/Documents/Experiments/Subjectivity_Quantified/')
    lm = Word2Vec.load('./LanguageModels/wikipedia/model1')

    results = []
    
    for lmd in lang_models:
        print('Training classifier on %s' % lmd)
        print()
        lm2= Word2Vec.load('./LanguageModels/'+lmd+'/model1')
        dataset = Concat_Sentence_Polarity_Dataset('/home/jbarnes/Downloads/rt-polaritydata/rt-polaritydata/', lm, lm2)
        classifier = create_classifier(600)
        test_dataset(dataset, classifier)

def test_600_wikipedia(vec_file):
    lm = WordVecs(vec_file)
    dataset = Sentence_Polarity_Dataset('/home/jbarnes/Downloads/rt-polaritydata/rt-polaritydata/',
                                        lm)
    classifier = create_classifier(600)
    history = classifier.fit(dataset._Xtrain, dataset._ytrain,
                             validation_data=[dataset._Xdev, dataset._ydev],
                      nb_epoch=10, verbose=0, batch_size=32)
    pred = classifier.predict(dataset._Xtest)
    mm = MyMetrics(dataset._ytest, pred, label_names=['N', 'P'], labels=[0,1],
                   average='macro')
    prec, rec, f1 = mm.get_scores()
    acc = mm.accuracy()
    print('\n\tprecision: {0}\n\trecall: {1}\n\tf1: {2}\n\tacc: {3}\n\n'.format((prec,
                                                                   rec,
                                                                   f1,
                                                                   acc)))
    

if __name__ == "__main__":

    
    
    models = ['extracted_subjectivity_corpora/multiun/high_precision/',
              'extracted_subjectivity_corpora/multiun/high_recall',
              'extracted_subjectivity_corpora/europarl/high_precision',
              'extracted_subjectivity_corpora/europarl/high_recall',
              'wikipedia']
    test_models(models)
    test_600_wikipedia('/home/jbarnes/mini_desktop/600vecs.txt')
