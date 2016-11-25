import os

def get_label_and_segments(label_text):
    info, label = label_text.split()
    corpus, sub_corpus, begin, end = info.split('_')
    return corpus, sub_corpus, int(begin), int(end), label


def get_subj_sents(labels, corpus_name):
    corpus = open(corpus_name).read()
    labels = open(labels).readlines()
    subj_sents = []
    for label_text in labels:
        corpusname, sub_corpus, begin, end, label = get_label_and_segments(label_text)
        if label == 'subj':
            subj_sents.append(corpus[begin:end])
    return subj_sents

def get_polar_sents(labels, corpus_name):
    corpus = open(corpus_name).read()
    labels = open(labels).readlines()
    polar_sents = []
    for label_text in labels:
        corpusname, sub_corpus, begin, end, label = get_label_and_segments(label_text)
        if label == 'positive' or label == 'negative':
            polar_sents.append(corpus[begin:end])
    return polar_sents


def get_high_precision_subj_corpus(corpus_name):
    docs = os.listdir('../opinionfinderv2.0/database/docs/'+ corpus_name)
    all_subj_sents = []
    for doc in docs:
        corpus = '../opinionfinderv2.0/database/docs/'+ corpus_name +'/'+doc
        subj_labels = '../opinionfinderv2.0/database/auto_anns/' + corpus_name +'/'+doc +'/sent_rule.txt'
        subj_sents = get_subj_sents(subj_labels, corpus)
        all_subj_sents.extend(subj_sents)
    return all_subj_sents

def get_high_recall_subj_corpus(corpus_name):
    docs = os.listdir('../opinionfinderv2.0/database/docs/'+ corpus_name)
    all_subj_sents = []
    for doc in docs:
        corpus = '../opinionfinderv2.0/database/docs/'+ corpus_name +'/'+doc
        subj_labels = '../opinionfinderv2.0/database/auto_anns/' + corpus_name +'/'+doc +'/sent_subj.txt'
        subj_sents = get_subj_sents(subj_labels, corpus)
        all_subj_sents.extend(subj_sents)
    return all_subj_sents

def get_polarity_corpus(corpus_name):
    docs = os.listdir('../opinionfinderv2.0/database/docs/'+ corpus_name)
    all_polar_sents = []
    for doc in docs:
        corpus = '../opinionfinderv2.0/database/docs/'+ corpus_name +'/'+doc
        polar_labels = '../opinionfinderv2.0/database/auto_anns/' + corpus_name +'/'+doc +'/exp_polarity.txt'
        polar_sents = get_polar_sents(polar_labels, corpus)
        all_polar_sents.extend(polar_sents)
    return all_polar_sents

def write_corpus(outfile, corpus):
    with open(outfile, 'w') as out:
        for line in corpus:
            out.write(line + '\n')

def extract_subj_corpora(corpus_name):
    high_recall_corpus = get_high_recall_subj_corpus(corpus_name)
    high_precision_corpus = get_high_precision_subj_corpus(corpus_name)

    high_recall_out = '../opinionfinderv2.0/database/docs/extracted_subjectivity_corpora/'+corpus_name+'/high_recall.txt'
    write_corpus(high_recall_out, high_recall_corpus)

    high_precision_out = '../opinionfinderv2.0/database/docs/extracted_subjectivity_corpora/'+corpus_name+'/high_precision.txt'
    write_corpus(high_precision_out, high_precision_corpus)



if __name__ == '__main__':

    corpora = ['europarl', 'multiun']

    for corpus in corpora:
        extract_subj_corpora(corpus)
