import os
import subprocess
import re
from gensim.models import word2vec
import matplotlib.pyplot as plt
import numpy as np


def run_opinionfinder(doc_list):

    """Takes a list of documents to be processessed in the form
       of a txt document with the name of one file per line.
       Creates folders with the output of opinionfinder."""

    for doc in open(doc_list).readlines():
        call = 'java -Xmx2g -classpath ./lib/weka.jar:./lib/stanford-postagger.jar:opinionfinder.jar opin.main.RunOpinionFinder %s -s -r preprocessor,cluefinder,rulebasedclass,subjclass,polarityclass' % doc
        subprocess.call(call, shell=True)

def get_sub(sents):
    """Takes the output list of opinionfinder and
    returns the count of subjective sentences"""

    num_subj = 0
    for sent in sents:
        if sent.strip().endswith('subj'):
            num_subj += 1
    return num_subj

def subjectivity(sents):
    """Returns the number of subjective sentences
    divided by the total number of sentences"""
    return get_sub(sents) / len(sents)

#TODO combine get_pol and polarity
def get_pol(sents):
    """Returns the number of polar sentences
    in a group of sentences"""
    num_pol = 0
    for sent in sents:
        if sent.strip().endswith('positive'):
            num_pol += 1
        elif sent.strip().endswith('negative'):
            num_pol += 1
    return num_pol

def polarity(sents):
    """Returns the number of polar sentences
    divided by the total number of sentences"""
    return get_pol(sents) / len(sents)

def get_auto_anns(filename):
    """Due to the database structure, we need to change
    the name of the files in the doclist."""
    return re.sub('docs', 'auto_anns', filename)

def calculate_high_recall_file(file_name):
    """calculates opinion finder's high recall results"""
    file_name = get_auto_anns(file_name)
    document = open(file_name+'/sent_subj.txt').readlines()
    return subjectivity(document)

def calculate_high_precision_file(file_name):
    """calculates opinion finder's high precision results"""
    file_name = get_auto_anns(file_name)
    document = open(file_name+'/sent_rule.txt').readlines()
    return subjectivity(document)

def calculate_polarity_file(file_name):
    """calculates opinion finder's polarity results"""
    file_name = get_auto_anns(file_name)
    document = open(file_name+'/exp_polarity.txt').readlines()
    return polarity(document)

def calculate_high_recall_doc_list(doc_list):
    """calculates the results for an entire doc list"""
    subj = 0
    tested = 0
    for file_name in open(doc_list).readlines():
        try:
            subj += calculate_high_recall_file(file_name.strip())
            tested += 1
        except FileNotFoundError:
            continue
    return subj / tested


def calculate_high_precision_doc_list(doc_list):
    """calculates the results for an entire doc list"""
    subj = 0
    tested = 0
    for file_name in open(doc_list).readlines():
        try:
            subj += calculate_high_precision_file(file_name.strip())
            tested += 1
        except FileNotFoundError:
            continue
    return subj / tested

def calculate_polarity_doc_list(doc_list):
    """calculates the results for an entire doc list"""
    pol = 0
    tested = 0
    for file_name in open(doc_list).readlines():
        try:
            pol += calculate_polarity_file(file_name.strip())
            tested += 1
        except FileNotFoundError:
            continue
    return pol / tested

def plot_subjectivity_of_corpora(high_recall_results, names, value_tested):
    width = .35
    y_pos = np.arange(len(names))
    plt.bar(y_pos, high_recall_results, width=width)
    plt.xticks(y_pos, names)
    plt.ylabel('Percent of %s' % value_tested)
    plt.title('%s Content of Various Corpora'% value_tested)
    plt.savefig('../Subjectivity_Results/%s.png' % value_tested, bbox_inches='tight')

def print_results(names, metrics, scores, out):
    out = open(out, 'w')
    out.write('Results of Subjectivity and Polarity Analysis\n')
    out.write('-'*80)
    out.write('\n')
    row_format = "{:>20}" * (len(names)+1)
    out.write(row_format.format('', * names)+'\n')
    for metric, row in zip(metrics, scores):
        out.write(row_format.format(metric, *['{:0.2f}'.format(name*100) for name in row])+'\n')

if __name__ == '__main__':

    doc_lists = ['multiun.doclist', 'opener.doclist', 'europarl.doclist', 'sentence_polarity.doclist']
    names = ['multiun', 'opener', 'europarl', 'sentence_polarity']

    high_recall = []
    high_precision = []
    pol = []


    for doc_list in doc_lists:
        recall_subj = calculate_high_recall_doc_list(doc_list)
        high_recall.append(recall_subj)
        print('The high recall subjectivity for %s is %.2f %%' % (doc_list, (recall_subj*100)))

        prec_subj = calculate_high_precision_doc_list(doc_list)
        high_precision.append(prec_subj)
        print('The high precision subjectivity for %s is %.2f %%' % (doc_list, (prec_subj*100)))

        p = calculate_polarity_doc_list(doc_list)
        pol.append(p)
        print('The polarity for %s is %.2f %%' % (doc_list, (p*100)))
        print()


    plot_subjectivity_of_corpora(high_recall, names, 'Subjectivity-High Recall')
    plot_subjectivity_of_corpora(high_precision, names, 'Subjectivity-High Precision')
    plot_subjectivity_of_corpora(pol, names, 'Polarity')
    
    scores = [high_recall, high_precision, pol]
    metrics = ['high recall subj.', 'high precision subj.', 'polarity']
    print_results(names, metrics, scores,
                  '../Subjectivity_Results/subjectivity_analysis_results.txt')
    

