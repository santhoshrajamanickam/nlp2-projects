from collections import defaultdict, Counter

class TextData:
    """
    Stores text data with additional attributes.
    :param fname: a path to a txt file
    """
    def __init__(self, fname):
        self._fname = fname
        self._data = []
        self._w2i = defaultdict(lambda: len(self._w2i))
        self._i2w = dict()
        self._counter = Counter()
        self._ntokens = 0    # number of tokens in dataset

        self._read()

    def _read(self):
        with open(self._fname, "r", encoding='utf8') as fh:
            for line in fh:
                tokens = line.strip().lower().split()
                self._data.append(tokens)
                self._counter.update(tokens)

        # Store number of tokens in the text
        self._ntokens = sum(self._counter.values())

        # Store the words in w2i in order of frequency from high to low
        for word, _ in self._counter.most_common(self._ntokens):
            self._i2w[self._w2i[word]] = word

    def __len__(self):
        """
        Number of tokens in the dataset
        """
        return self._ntokens

    @property
    def data(self):
        """
        The data as list of lists
        """
        return self._data

    @property
    def counter(self):
        """
        The word-counts as counter
        """
        return self._counter

    @property
    def vocab_size(self):
        """
        Number of words in the dataset
        """
        return len(self._w2i)

    @property
    def w2i(self):
        """
        Word to index dictionary
        Words are sorted in order of frequency from high to low
        """
        return self._w2i

    @property
    def i2w(self):
        """
        Inverse dictionary of w2i: index to words
        """
        return self._i2w
