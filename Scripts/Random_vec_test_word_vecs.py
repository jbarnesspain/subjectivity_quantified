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

def test_dataset(dataset, classifier):
    history = classifier.fit(dataset._Xtrain, dataset._ytrain, validation_data=[dataset._Xdev, dataset._ydev],
                      nb_epoch=10, verbose=1, batch_size=32)
    pred = classifier.predict(dataset._Xtest)
    acc = accuracy_score(dataset._ytest.argmax(axis=1), pred.argmax(axis=1))
    prec = precision_score(dataset._ytest.argmax(axis=1), pred.argmax(axis=1))
    rec =recall_score(dataset._ytest.argmax(axis=1), pred.argmax(axis=1))
    f1 = f1_score(dataset._ytest.argmax(axis=1), pred.argmax(axis=1))
    return acc, prec, rec, f1

class Random_Sentence_Polarity_Dataset(object):

    
    def __init__(self, DIR, model, one_hot=True,
                 dtype=np.float32, rep='avg'):

        self.not_found_in_backoff = []
        self.not_found_in_extracted = []
        self.rep = rep
        self.vector_size = model.vector_size
        self.random_vector = np.random.random((self.vector_size))

        dataneg = self.getMyData(os.path.join(DIR, 'neg.txt'), 0, model)
        datapos = self.getMyData(os.path.join(DIR, 'pos.txt'), 1, model)
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
                sent += self.random_vector
            except TypeError:
                sent += self.random_vector
      
        return sent/sent_length

    def getMyData(self, fname, label, model):
        if self.rep == 'avg':
            representation = self.averageWordVecs
        data= []
        for sent in codecs.open(fname, 'r', encoding='utf8', errors='replace'):
            data.append((representation(sent,model), label))
        return data

def test_models(lang_models):

    os.chdir('/home/jbarnes/Documents/Experiments/Subjectivity_Quantified/')

    results = []
    
    for lmd in lang_models:
        print('Training classifier on %s' % lmd)
        print()
        lang_model = Word2Vec.load('./LanguageModels/extracted_subjectivity_corpora/'+lmd+'/model1')
        dataset = Random_Sentence_Polarity_Dataset('/home/jbarnes/Downloads/rt-polaritydata/rt-polaritydata/', lang_model)
        classifier = create_classifier()
        acc, prec, rec, f1 = test_dataset(dataset, classifier)
        results.append((acc, prec, rec, f1))
        print("acc: {0}\nprec: {1}\nrec: {2}\nf1: {3}".format(acc, prec, rec, f1))
        print()
        print('-'*80)
        print()

if __name__ == "__main__":
    
    models = ['multiun/high_precision/', 'multiun/high_recall', 'europarl/high_precision', 'europarl/high_recall']
    test_models(models)
