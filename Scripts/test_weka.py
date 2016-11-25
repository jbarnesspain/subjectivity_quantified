import sys, os
import codecs
import re
import subprocess

def replace_quotes(sent):
    return re.sub('"', "'", sent)
    
def getdata(fname, label):
    data= []
    for sent in codecs.open(fname, 'r', encoding='utf8', errors='replace'):
        sent = replace_quotes(sent.strip())
        data.append((sent, label))
    return data

class Check_Dataset(object):
    def __init__(self, DIR):


        dataneg = getdata(os.path.join(DIR, 'neg.txt'), 'N')
        datapos = getdata(os.path.join(DIR, 'pos.txt'), 'P')
        self._traindata = dataneg[:int((len(dataneg) * .75))] + datapos[:int((len(datapos) * .75))]
        self._devdata = dataneg[int((len(dataneg) * .75)):int((len(dataneg) * .85))] + datapos[int((len(datapos) * .75)):int((len(dataneg) * .85))]
        self._testdata = dataneg[int((len(dataneg) * .85)):]+ datapos[int((len(datapos) * .85)):]
        self._Xtrain = [data for data, y in self._traindata]
        self._ytrain = [y for data, y in self._traindata]

        self._Xdev = [data for data, y in self._devdata]
        self._ydev = [y for data, y in self._devdata]

        self._Xtest = [data for data, y in self._testdata]
        self._ytest = [y for data,y in self._testdata]

	    
def convert_dataset_to_arff(data, outfile):
    header = """@relation Subjectivity_Quantified
@attribute tokens string
@attribute polarity {N, P}
@data
"""
    outfile.write(header)
    for sent, label in data:
        outfile.write('"'+sent+'",'+ label+'\n')
    outfile.close()

def run_weka(weka_script_directory):

    os.chdir(weka_script_directory) 
    call = 'bash run_weka.sh'
    subprocess.call(call, shell=True)


if __name__ == '__main__':

    sent_pol_dataset_dir = '/home/jbarnes/Documents/Experiments/Subjectivity_Quantified/sentence_polarity_dataset/'
    out_dir = '/home/jbarnes/Documents/Experiments/Subjectivity_Quantified/Weka_arffs'
    dataset = Check_Dataset(sent_pol_dataset_dir)

    out_train = open(out_dir + '/train.arff', 'w')
    out_dev = open(out_dir + '/dev.arff', 'w')
    out_test = open(out_dir + '/test.arff', 'w')

    convert_dataset_to_arff(dataset._traindata, out_train)
    convert_dataset_to_arff(dataset._devdata, out_dev)
    convert_dataset_to_arff(dataset._testdata, out_test)

    run_weka('/home/jbarnes/Documents/Experiments/Subjectivity_Quantified/Scripts')
    
