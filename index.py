from collections import Counter
import pandas as pd
import stop_words
import random


class NLP():

    def __init__(self):
        self.vocab = None

    def count_vectorizer(self, text, train=True, stop_word=None, view=False):


        lower_case_documents = []
        documents = text
        for i in documents:
            lower_case_documents.append(i.lower())

        if view:
            print('Step: Applying Lower Case.... Done\n')
        #     print(lower_case_documents)
        sans_punctuation_documents = []

        import string

        for i in lower_case_documents:
            punctuation = string.punctuation

            k = ""
            for j in i:
                if j not in punctuation:
                    k += j

            sans_punctuation_documents.append(k)

        if view:
            print('Step: Removed Punctuation....\n')
            print(sans_punctuation_documents)

        if stop_word == None:
            stop_word = list(stop_words.ENGLISH_STOP_WORDS)

        preprocessed_documents = []
        for i in sans_punctuation_documents:
            sentence = []
            for word in i.split():
                if word not in stop_word:
                    sentence.append(word)
            preprocessed_documents.append(sentence)

        if train != True:
            return preprocessed_documents

        if view:
            print('Step: Bag of Words... Done\n')
            print(preprocessed_documents)

        frequency_list = []
        from collections import Counter

        for i in preprocessed_documents:
            frequency_list.append(dict(Counter(i)))

        if view:
            print('Step: Frequency of words... Done\n')

        # often called as vocabulary
        all_words = list(set([j for i in preprocessed_documents for j in i]))

        for doc in frequency_list:
            for word in all_words:
                if word not in list(doc.keys()):
                    doc[word] = 0
        df = pd.DataFrame(frequency_list)
        df = df[sorted(list(df.columns))]

        self.vocab = df.columns.to_list()

        if view:
            print('Step: Count vectorizer... Done\n')
            print(df.head())
        return df
nlp = NLP()


class NaiveBayes():

    def __init__(self, text, label):
        self.text = text
        self.label = label
        self.cond_probs = {}
        self.features = []
        self.classes = []
        self.class_prob = {}

    def fit(self, view=False):

        text = self.text
        label = self.label

        bow = nlp.count_vectorizer(text)

        self.features = bow.columns.to_list()

        if view:
            print('Your BoW is:\n', bow)

        classes = label

        self.classes = list(Counter(classes).keys())

        bow['out'] = classes
        bow_class = bow.groupby(by='out', axis=0)

        if view:
            print('Your BoW is testing:\n')

        # count of each class examples
        counts = bow_class.count()
        if view:
            print(counts)
            print

        # used for prediction
        class_prob = counts / counts.sum(axis=0)
        class_prob = dict(class_prob.mean(axis=1))
        self.class_prob = class_prob
        if view:
            print(class_prob)

        # count of each word on each class
        self.count_words_class = bow_class.sum()

        # find prob of word in each class.... no. of that word in class / total word in class
        prob_w_c = (bow_class.sum() + 1) / (counts )
        if view:
            print("chenck the testing prob")
            print(prob_w_c)
        # find p(word/class)

        prob_w_c = round(prob_w_c, 5)
        self.cond_probs = prob_w_c
        if view:
            print(prob_w_c)

    def classes_(self):
        """
        A method to see all classes counts for each word.
        """
        return self.count_words_class

    def predict(self, example):

        txt = nlp.count_vectorizer(example, train=False)
        words = dict(Counter(txt[0]))

        vocab = self.features
        classes = self.classes
        class_prob = self.class_prob
        p = self.cond_probs



        prob_zero = class_prob['0']

        prob_one = class_prob['1']


        for w in words.keys():

            if w in vocab:
                prob_zero = prob_zero * p[w][0]

                prob_one = prob_one * p[w][1]

            else:

                prob_zero = prob_zero * 10
                prob_one = prob_one * 10

        if (prob_zero < prob_one):
            return 1
        else:
            return 0


f = open("dataset_NB.txt", "r")
t = f.read().splitlines()
f.close()
# print(np.shape(t))
a = random.sample(t, len(t))

X = []
Y = []
all_txt = []
classes = []
for i in a:
    X.append(i[:-1])
    Y.append(i[-1])

append_list_x = []
append_list_y = []
train_range_1 = 0
for k in range(7):

    train_range_2=train_range_1 + 1000//7

    append_list_x = X[0:train_range_1]
    append_list_y = Y[0:train_range_1]

    all_txt = X[train_range_2:] + append_list_x
    classes = Y[train_range_2:] + append_list_y
    testX = X[train_range_1:train_range_2]
    testY = Y[train_range_1:train_range_2]
    nb = NaiveBayes(all_txt, classes)
    nb.fit()

    count_total = 0
    count_metrics = 0
    res = 0

    for i in range(0, len(testY)):
        testY[i] = int(testY[i])

    for i in testX:
        a = 0

        count_total = count_total + 1
        res = nb.predict([i])

        if res == testY[a]:
            count_metrics = count_metrics + 1
        else:
            count_metrics = count_metrics + 0
        a = a + 1
    accuracy_final = count_metrics / count_total
    print("Accuracy of fold %d is"%k,end = "")
    print(accuracy_final)
    train_range_1 = train_range_1 + 1000//7
    




