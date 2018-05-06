# FengTing Liao 2018
#
# Part of the code comes from - https://github.com/uclmr/fakenewschallenge
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Import relevant packages and modules
import io
from csv import DictReader
from csv import DictWriter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
import tensorflow as tf

import keras as K
from keras.models import Model, Sequential
from keras.layers import Input, Add, Activation, Dense, BatchNormalization, Dropout, LeakyReLU
from keras.utils import np_utils, plot_model
from keras.initializers import glorot_normal
from keras import regularizers


# Initialise global variables
label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
label_ref_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]


# Define data class
class FNCData:

    """

    Define class for Fake News Challenge data

    """

    def __init__(self, file_instances, file_bodies):

        # Load data
        self.instances = self.read(file_instances)
        bodies = self.read(file_bodies)
        self.heads = {}
        self.bodies = {}

        # Process instances
        for instance in self.instances:
            if instance['Headline'] not in self.heads:
                head_id = len(self.heads)
                self.heads[instance['Headline']] = head_id
            instance['Body ID'] = int(instance['Body ID'])

        # Process bodies
        for body in bodies:
            self.bodies[int(body['Body ID'])] = body['articleBody']

    def read(self, filename):

        """
        Read Fake News Challenge data from CSV file

        Args:
            filename: str, filename + extension

        Returns:
            rows: list, of dict per instance

        """

        # Initialise
        rows = []

        # Process file
        with open(filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)

        return rows


# Define relevant functions
def pipeline_train(train, test, lim_unigram):

    """

    Process train set, create relevant vectorizers

    Args:
        train: FNCData object, train set
        test: FNCData object, test set
        lim_unigram: int, number of most frequent words to consider

    Returns:
        train_set: list, of numpy arrays
        train_stances: list, of ints
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    """

    # Initialise
    heads = []
    heads_track = {}
    bodies = []
    bodies_track = {}
    body_ids = []
    id_ref = {}
    train_set = []
    train_stances = []
    cos_track = {}
    test_heads = []
    test_heads_track = {}
    test_bodies = []
    test_bodies_track = {}
    test_body_ids = []
    head_tfidf_track = {}
    body_tfidf_track = {}

    # Identify unique heads and bodies
    for instance in train.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            heads.append(head)
            heads_track[head] = 1
        if body_id not in bodies_track:
            bodies.append(train.bodies[body_id])
            bodies_track[body_id] = 1
            body_ids.append(body_id)

    for instance in test.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in test_heads_track:
            test_heads.append(head)
            test_heads_track[head] = 1
        if body_id not in test_bodies_track:
            test_bodies.append(test.bodies[body_id])
            test_bodies_track[body_id] = 1
            test_body_ids.append(body_id)

    # Create reference dictionary
    for i, elem in enumerate(heads + body_ids):
        id_ref[elem] = i

    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=lim_unigram, stop_words=stop_words)
    bow = bow_vectorizer.fit_transform(heads + bodies)  # Train set only

    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only

    tfidf_vectorizer = TfidfVectorizer(max_features=lim_unigram, stop_words=stop_words).\
        fit(heads + bodies + test_heads + test_bodies)  # Train and test sets

    # Process train set
    for instance in train.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        head_tf = tfreq[id_ref[head]].reshape(1, -1)
        body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
        if head not in head_tfidf_track:
            head_tfidf = tfidf_vectorizer.transform([head]).toarray()
            head_tfidf_track[head] = head_tfidf
        else:
            head_tfidf = head_tfidf_track[head]
        if body_id not in body_tfidf_track:
            body_tfidf = tfidf_vectorizer.transform([train.bodies[body_id]]).toarray()
            body_tfidf_track[body_id] = body_tfidf
        else:
            body_tfidf = body_tfidf_track[body_id]
        if (head, body_id) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(head, body_id)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(head, body_id)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        train_set.append(feat_vec)
        train_stances.append(label_ref[instance['Stance']])

    return train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer


def pipeline_test(test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):

    """

    Process test set

    Args:
        test: FNCData object, test set
        bow_vectorizer: sklearn CountVectorizer
        tfreq_vectorizer: sklearn TfidfTransformer(use_idf=False)
        tfidf_vectorizer: sklearn TfidfVectorizer()

    Returns:
        test_set: list, of numpy arrays

    """

    # Initialise
    test_set = []
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    # Process test set
    for instance in test.instances:
        head = instance['Headline']
        body_id = instance['Body ID']
        if head not in heads_track:
            head_bow = bow_vectorizer.transform([head]).toarray()
            head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
            head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)
            heads_track[head] = (head_tf, head_tfidf)
        else:
            head_tf = heads_track[head][0]
            head_tfidf = heads_track[head][1]
        if body_id not in bodies_track:
            body_bow = bow_vectorizer.transform([test.bodies[body_id]]).toarray()
            body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
            body_tfidf = tfidf_vectorizer.transform([test.bodies[body_id]]).toarray().reshape(1, -1)
            bodies_track[body_id] = (body_tf, body_tfidf)
        else:
            body_tf = bodies_track[body_id][0]
            body_tfidf = bodies_track[body_id][1]
        if (head, body_id) not in cos_track:
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            cos_track[(head, body_id)] = tfidf_cos
        else:
            tfidf_cos = cos_track[(head, body_id)]
        feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
        test_set.append(feat_vec)

    return test_set

# Loading up data

def load_data(file_train_instances = "../DATA/train_stances.csv"
              ,file_train_bodies = "../DATA/train_bodies.csv"
              ,file_test_instances = "../DATA/test_stances_unlabeled.csv"
              ,file_test_bodies = "../DATA/test_bodies.csv"
              ,train_dev_split = 0.8
              ,lim_unigram = 5000
              ,random_seed = 0):
    # Load data sets
    raw_train = FNCData(file_train_instances, file_train_bodies)
    raw_test = FNCData(file_test_instances, file_test_bodies)
    n_train = len(raw_train.instances)
    print('Raw data loaded.... number of data points = %d' % (n_train))
    # Process data sets
    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    feature_size = len(train_set[0])
    test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)

    print('Initial setup for training dataset complete....')
    dic = {}
    # Setting up training data
    train_data = np.array(train_set)
    ## Let labels become one-hot representation
    tmp = np.array(train_stances)
    train_label = np.zeros((tmp.shape[0], 4))
    train_label[np.arange(tmp.shape[0]), tmp] = 1
    
    train_sparse = coo_matrix(train_data)
    train_data, train_sparse, train_label = shuffle(train_data, train_sparse, train_label, random_state=random_seed)
    
    # Setting up test data
    test_data = np.array(test_set)
    
    idx_split = int(n_train*train_dev_split)
    dic['train_x'] = train_data[:idx_split,:]
    dic['train_y'] = train_label[:idx_split,:]
    dic['dev_x'] = train_data[idx_split:,:]
    dic['dev_y'] = train_label[idx_split:,:]
    dic['test_x'] = test_data
    print('Finished setting up training set. Number of train/dev/test data points = %d/%d/%d' 
          % (dic['train_x'].shape[0], dic['dev_x'].shape[0], dic['test_x'].shape[0]))
    return dic
    

# Setup model - batchnorm before Activation
# 
def set_model_NN(input_output_shape = (12, 4), 
                 hidden_nodes=[30, 30], 
                 activation='relu',
                 loss = 'categorical_crossentropy',
                 metrics = ['categorical_accuracy'],
                 initializer = 'normal',
                 reg_l2 = 0.01,
                 epsilon=1e-6):
    n_input = input_output_shape[0]
    n_output = input_output_shape[1]
    model = Sequential()
    
    # For the first layer, activation needs to be set directly inside a dense layer
    if activation == 'leakyrelu':
        model.add(Dense(hidden_nodes[0], input_dim=n_input, kernel_initializer=initializer, activation='linear'
                       ,kernel_regularizer=regularizers.l2(reg_l2)))
        model.add(BatchNormalization(axis=1, name='bn0',epsilon=epsilon))
        model.add(LeakyReLU(alpha=0.3))
    else:
        model.add(Dense(hidden_nodes[0], input_dim=n_input, kernel_initializer=initializer
                       ,kernel_regularizer=regularizers.l2(reg_l2)))
        model.add(BatchNormalization(axis=1, name='bn0',epsilon=epsilon))
        model.add(Activation(activation=activation))
    
    # Set up the hidden layers
    for i, node in enumerate(hidden_nodes[1:]):
        if activation == 'leakyrelu':
            model.add(Dense(hidden_nodes[i+1], kernel_initializer=initializer, activation='linear'
                           ,kernel_regularizer=regularizers.l2(reg_l2)))    
            model.add(BatchNormalization(axis=1, name='bn'+str(i+1),epsilon=epsilon))
            model.add(LeakyReLU(alpha=0.3))
        else:
            model.add(Dense(hidden_nodes[i+1], kernel_initializer=initializer
                           ,kernel_regularizer=regularizers.l2(reg_l2)))    
            model.add(BatchNormalization(axis=1, name='bn'+str(i+1),epsilon=epsilon))
            model.add(Activation(activation))
        #print(i, node)
    
    model.add(Dense(n_output, activation='linear'))
    model.compile(loss=loss, optimizer='adam', 
                  metrics=metrics)
    return model

# Plotting for TSNE scattering plot
def plot_tsne_scattering(X, Y, dic_labels):    
    fig, ax = plt.subplots(figsize=(10,10))
    #fig.figure(figsize=(8,8))
    colors = ['red', 'green', 'blue','black']
    for i, color in enumerate(colors):
        idx = Y_embedded==i
        npts = np.argwhere(idx).shape[0]
        ax.scatter(X_embedded[idx,0], X_embedded[idx,1], c=color, label=str(npts)+' '+dic_labels[i],
                   alpha=0.5, edgecolors='none')
    
    ax.set_xlabel('arbitrary', fontsize='x-large')
    ax.set_ylabel('arbitrary', fontsize='x-large')
    ax.tick_params(axis='both', which='major', labelsize='x-large')
    ax.legend(fontsize='x-large')
    ax.grid(True)
    plt.show()
