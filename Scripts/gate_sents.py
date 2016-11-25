class GateReader(object):
    def __init__(self, filename):
        self.file = open(filename).readlines()
        self.tokens = self.get_tokens(self.file)
        self.sents = self.get_sentences(self.file, self.tokens)


    def get_sentences(self, gate_output, all_tokens):
        sents = {}
        for line in gate_output:
            if line.split()[2] == 'GATE_Sentence':
                num, start, end = self.sentence_to_Gate(line)
                sents[num] = Sentence(start, end, all_tokens)
        return sents

    def sentence_to_Gate(self, sent):
        num, start_end, _ = sent.split()
        start, end = start_end.split(',')
        num = eval(num)
        start = eval(start)
        end = eval(end)
        return num, start, end


    def get_tokens(self, gate_output):
        tokens = {}
        for line in gate_output:
            if line.split()[2] == 'GATE_Token':
                num, start, end, string, lemma, cat = self.token_to_Gate(line)
                tokens[(start, end)] = Token(start, end, string, lemma, cat)
        return tokens

    def token_to_Gate(self, sent):
        num, start_end, _, string, lemma, category = sent.split()
        start, end = start_end.split(',')
        num = eval(num)
        start = eval(start)
        end = eval(end)
        string = string.split('"')[1]
        lemma = lemma.split('"')[1]
        category = category.split('"')[1]
        return num, start, end, string ,lemma, category

class Sentence():
    def __init__(self, start_char, end_char, all_tokens):
        self.start_char = start_char
        self.end_char = end_char
        self.tokens = self.get_sentence_tokens(all_tokens)

    def __str__(self):
        return 'char %i: char %i' %(self.start_char, self.end_char)

    def print_sent(self):
        return self.__str__()

    def get_sentence_tokens(self, tokens):
        
        sentence_tokens = []
        for start, end in tokens.keys():
            if start >= self.start_char and end <= self.end_char:
                sentence_tokens.append(tokens[(start, end)])
        return sentence_tokens

class Token():
    def __init__(self, start_char, end_char, string, lemma, category):
        self.start_char = start_char
        self.end_char = end_char
        self.string = string
        self.lemma = lemma
        self.category = category

    def __str__(self):
        return ('char %i: char %i\nstring="%s"\nlemma="%s"\ncategory="%s"'
                %(self.start_char, self.end_char, self.string, self.lemma,
                  self.category))

    def print_token(self):
        print(self.__str__())

