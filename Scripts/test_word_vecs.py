import sys, os
sys.path.append('/home/jbarnes/Documents/Experiments/BilBowa_Experiments')
from Datasets import *
from MyMetrics import *
from scipy.spatial.distance import cosine
import numpy as np
import codecs
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

def averageWordVecs(sentence,model):
    """Returns a vector which is the
    averate of all of the vectors in
    the sentence"""
    sent = np.array(np.zeros((model.vector_size)))
    words = sentence.split()
    sent_length = len(words)
    for w in words:
        try:
            sent += model[w]
        except KeyError:
            sent_length -= 1
            pass
        except TypeError:
            sent_length -= 1
            pass
        except ValueError:
            sent_length -= 1
            pass
    return sent/sent_length


def getMyData(fname, label, model, representation=averageWordVecs):
    data= []
    for sent in codecs.open(fname, 'r', encoding='utf8', errors='replace'):
        data.append((representation(sent,model), label))
    return data

class Sentence_Polarity_Dataset(object):
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

    def __init__(self, DIR, model, one_hot=True,
                 dtype=np.float32, rep=averageWordVecs):


        dataneg = getMyData(os.path.join(DIR, 'neg.txt'), 1, model, representation=rep)
        datapos = getMyData(os.path.join(DIR, 'pos.txt'), 2, model, representation=rep)
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



def create_classifier(output_dim=2):
    clf = Sequential()
    clf.add(Dense(input_dim=300, output_dim=800, activation='relu'))
    clf.add(Dropout(.5))
    clf.add(Dense(input_dim=800, output_dim=800, activation='relu'))
    clf.add(Dropout(.5))
    clf.add(Dense(input_dim=800, output_dim=300, activation='relu'))
    clf.add(Dense(input_dim=300, output_dim=output_dim, activation='softmax'))
    clf.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    return clf

def create_svm():
    clf = LinearSVC()
    return clf


def test_dataset(dataset, classifier, svm=True):
    if svm == False:
        history = classifier.fit(dataset._Xtrain, dataset._ytrain, validation_split=.1,
                    nb_epoch=10, verbose=0, batch_size=32)
        pred = classifier.predict(dataset._Xtest)
        mm = MyMetrics(dataset._ytest, pred, label_names=['N', 'P'], labels=[0,1])
    else:
        classifier.fit(dataset._Xtrain, dataset._ytrain)
        pred = classifier.predict(dataset._Xtest)
        mm = MyMetrics(dataset._ytest, pred, one_hot=False, label_names=['N', 'P'], labels=[1,2])
    mm.print_metrics()

def test_all(lang_models):

    
    for lmd in lang_models:
        print('Training classifier on %s' % lmd)
        print()
        lang_model = Word2Vec.load('../LanguageModels/'+lmd+'/model1')
        dataset = Sentence_Polarity_Dataset('../rt-polaritydata/rt-polaritydata/',
                                            lang_model,
                                            one_hot=False)
        #dataset = English_Dataset(lang_model)
        classifier = create_svm()
        test_dataset(dataset, classifier)


def test_all_small_models(lang_models):

    
    for lmd in lang_models:
        print('Training classifier on %s' % lmd)
        print()
        lang_model = Word2Vec.load('../LanguageModels/'+lmd+'/small_model')
        dataset = Sentence_Polarity_Dataset('../rt-polaritydata/rt-polaritydata/', lang_model)
        #dataset = English_Dataset(lang_model)
        classifier = create_classifier(2)
        test_dataset(dataset, classifier)
      

class GloveVecs(object):

    def __init__(self, file, vector_size):
        self.word_to_vec = self.read_vecs(file)
        self.vector_size = vector_size
        self.vocab_length = len(self.word_to_vec)

    def __getitem__(self,y):
        return self.word_to_vec.get(y)
    
    def read_vecs(self, file):
        txt = open(file).readlines()
        word_to_vec = {}
        lines_read = 0
        for item in txt:
            try:
                split = item.split()
                word, vec = split[0], np.array(split[1:], dtype='float32')
                word_to_vec[word] = vec
                lines_read += 1
                if lines_read % 100 == 0:
                    self.drawProgressBar((lines_read/len(txt)))
            except NameError:
                continue
            except SyntaxError:
                continue
            except ValueError:
                continue

        return word_to_vec
    
    def drawProgressBar(self, percent, barLen = 20):
        sys.stdout.write("\r")
        progress = ""
        for i in range(barLen):
            if i < int(barLen * percent):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
        sys.stdout.flush()

if __name__ == '__main__':

    print()
    print('FULL SIZED CORPORA')
    print()
    
    lang_models = ['europarl', 'multiun', 'wikipedia', 'opener', 'open_subtitles', 'sentence_polarity']
    test_all(lang_models)

    print()
    print('10 MILLION WORD VERSION')
    print()

    lang_models2 = ['europarl','multiun', 'wikipedia']
    test_all_small_models(lang_models2)

    print()
    print('wikipedia + others')
    print()

    lang_models3 = ['extracted_subjectivity_corpora/europarl/high_recall',
                    'extracted_subjectivity_corpora/europarl/high_precision',
                    'extracted_subjectivity_corpora/multiun/high_recall',
                    'extracted_subjectivity_corpora/multiun/high_precision',
                    'extracted_subjectivity_corpora/wikipedia/20_million_word']
    test_all(lang_models3)

    print()
    print('Subjectivity extracted wikipedia')
    print()

    lang_models4 = ['extracted_subjectivity_corpora/wikipedia/high_precision',
                    'extracted_subjectivity_corpora/wikipedia/high_recall']
    test_all(lang_models4)

    """
    glove_vecs = GloveVecs('/home/jbarnes/Downloads/glove.6B/glove.6B.300d.txt', 300)
    dataset = English_Dataset(glove_vecs, rep=averageWordVecs)
    clf = create_classifier()
    acc, prec, rec, f1 = test_dataset(dataset5, clf)
    print("acc: {0}\nprec: {1}\nrec: {2}\nf1: {3}".format(acc, prec, rec, f1))
    
    #model = Word2Vec.load_word2vec_format('/home/jbarnes/Documents/WikiCorpora/GoogleNews-vectors-negative300.bin.gz', binary=True)
    model = Word2Vec.load('/home/jbarnes/Documents/Experiments/Subjectivity_Quantified/LanguageModels/wikipedia/small_model')
    dataset = Sentence_Polarity_Dataset('/home/jbarnes/Documents/Experiments/Subjectivity_Quantified/sentence_polarity_dataset/', model)
    classifier = create_classifier()
    acc, prec, rec, f1 = test_dataset(dataset, classifier)
    print("acc: {0}\nprec: {1}\nrec: {2}\nf1: {3}".format(acc, prec, rec, f1))"""
