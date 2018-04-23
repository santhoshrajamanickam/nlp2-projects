english_train = 'training/hansards.36.2.e'
french_train = 'training/hansards.36.2.f'


class IBM1():

    def __init__(self):
        self.english_training = []
        self.french_training = []
        self.english_voc = set()
        self.french_voc = set()
        self.english_indices = dict()
        self.english_words = dict()
        self.french_indices = dict()
        self.french_words = dict()


    def read_data(self, english_train, french_train):
        print('Start reading data...')

        e = open(english_train, 'r', encoding='utf8')
        for i, line in enumerate(e):
            sentence = line.split()
            # add null word to each sentence
            sentence = ['NULL'] + sentence
            self.english_training.append(sentence)
            # add words to vocabulary
            self.english_voc.update(sentence)
        e.close()


        f = open(french_train, 'r', encoding='utf8')
        for i, line in enumerate(f):
            sentence = line.split()
            self.french_training.append(sentence)
            self.french_voc.update(sentence)
        f.close()

        for index, e in enumerate(self.english_voc):
            self.english_indices = index
            self.english_words[index] = e

        for index, f in enumerate(self.french_voc):
            self.french_indices[f] = index
            self.french_words[index] = f


        print('Done with reading data!')
