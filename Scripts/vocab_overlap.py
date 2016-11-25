import nltk
import os

class Vocab_Overlap():
    def __init__(self, sentence_polarity_dir):
        pos_train = open(sentence_polarity_dir+'/train/pos.txt').readlines()
        neg_train = open(sentence_polarity_dir+'/train/neg.txt').readlines()
        pos_test = open(sentence_polarity_dir+'/test/pos.txt').readlines()
        neg_test = open(sentence_polarity_dir+'/test/neg.txt').readlines()
        
        train = pos_train + neg_train
        test = pos_test + neg_test

        train_tokenized = [nltk.word_tokenize(sent) for sent in train]
        test_tokenized = [nltk.word_tokenize(sent) for sent in test]
        
        self.train_sent_pol_test_vocab = sorted(set([w.lower() for sent in
                                               train_tokenized for w in sent]))
        self.test_sent_pol_test_vocab = sorted(set([w.lower() for sent in
                                               test_tokenized for w in sent]))

    def lexical_overlap(self, second_corpus_vocab):
        in_both_train = [w for w in self.train_sent_pol_test_vocab if w in
                   second_corpus_vocab]
        train_overlap = len(in_both_train) / len(self.train_sent_pol_test_vocab)
        in_both_test = [w for w in self.test_sent_pol_test_vocab if w in
                   second_corpus_vocab]
        test_overlap = len(in_both_test) / len(self.test_sent_pol_test_vocab)
        return train_overlap, test_overlap

    def lexical_diversity(self, vocab, tokens):
        return len(vocab) / len(tokens)

    def get_corpus_statistics(self, corpus_file, num_tokens=None, multiple_files=False):
        if num_tokens is not None:
            if multiple_files==False:
                raw = open(corpus_file).read()
                tokenized = nltk.word_tokenize(raw)
            else:
                tokenized_sents = []
                for fname in os.listdir(corpus_file):
                    sub_dir = os.path.join(corpus_file, fname)
                    for fname in os.listdir(sub_dir):
                        text_file = os.path.join(sub_dir, fname)
                        raw = open(text_file).read()
                        tokenized_sents.append(nltk.word_tokenize(raw))
                tokenized = [w for sent in tokenized_sents for w in sent]
                
            
            full_corpus = tokenized
            sample_corpus = tokenized[:num_tokens]

            full_corpus_vocab = sorted(set([w.lower() for w in full_corpus]))
            sample_corpus_vocab = sorted(set([w.lower() for w in sample_corpus]))
            lexical_diversity = self.lexical_diversity(sample_corpus_vocab,
                                                       sample_corpus)
            train_lexical_overlap, test_lexical_overlap = \
                                   self.lexical_overlap(full_corpus_vocab)
            self.print_stats(full_corpus_vocab,
                        lexical_diversity,
                        train_lexical_overlap,
                        test_lexical_overlap,
                        sample_corpus_vocab)
        else:
            if multiple_files==False:
                raw = open(corpus_file).read()
                corpus = nltk.word_tokenize(raw)
            else:
                tokenized_sents = []
                for fname in os.listdir(corpus_file):
                    sub_dir = os.path.join(corpus_file, fname)
                    for fname in os.listdir(sub_dir):
                        text_file = os.path.join(sub_dir, fname)
                        raw = open(text_file).read()
                        tokenized_sents.append(nltk.word_tokenize(raw))
                corpus = [w for sent in tokenized_sents for w in sent]

            corpus_vocab = sorted(set([w.lower() for w in corpus]))
            lexical_diversity = self.lexical_diversity(corpus_vocab, corpus)
            lexical_overlap = self.lexical_overlap(corpus_vocab)
            self.print_stats(corpus_vocab,
                        lexical_diversity,
                        train_lexical_overlap,
                        test_lexical_overlap)
            

    def print_stats(self, corpus_vocab, lexical_diversity, train_lexical_overlap,
                    test_lexical_overlap, sample_corpus_vocab=None):
            print('Length of corpus vocabulary: {0}'.format(len(corpus_vocab)))
            print('Lexical_diversity measure: {0}'.format((lexical_diversity)))
            print('Lexical overlap with train: {0}'.format(train_lexical_overlap))
            print('Lexical overlap with test: {0}'.format(test_lexical_overlap))
            if sample_corpus_vocab is not None:
                print('Length of sample corpus vocabulary used to examine lexical diversity: {0}'.format(
                    len(sample_corpus_vocab)))
                
        
