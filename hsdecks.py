import gzip
import io
import os
import platform
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import hearthstone.cardxml
import nltk
import numpy as np
import tensorflow as tf
from hearthstone.enums import CardType, GameTag
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
import string

metastats_url = "https://s3.amazonaws.com/metadetector/metaDecks.xml.gz"
carddefs_url = "https://github.com/HearthSim/hsdata/raw/master/CardDefs.xml"

metadecks_path = 'metaDecks.xml'
carddefs_path = 'CardDefs.xml'

MANA_COST_RANGE = len(range(0, 11))

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

def download_decks(url, filename_out):
    response = urllib.request.urlopen(url)
    compressed_file = io.BytesIO(response.read())
    decompressed_file = gzip.GzipFile(fileobj=compressed_file)

    with open(filename_out, 'wb') as outfile:
        outfile.write(decompressed_file.read())

def download_carddefs(url, filename_out):
    response = urllib.request.urlopen(url)

    with open(filename_out, 'wb') as outfile:
        outfile.write(response.read())


class Deck():
    def __init__(self, id, hero_class):
        self.id = id
        self.cards = []
        self.hero_class = hero_class

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def clean_text(text):
    text = cleanhtml(text)
    text = text.replace('\n', ' ')
    text = text.replace('_', ' ')
    # text = text.replace('\'s', '')
    # text = text.replace('@', '')
    text = text.replace('\'', '')
    text = text.replace('\"', '')
    text = text.replace(']', ' ')
    # text = text.replace('\'re', '')

    text = text.translate(str.maketrans('', '', '@.,:;[]()!’-'))

    return text


def replace_common_text(text):
    p = re.compile(r"([0-9]\/[0-9])\s([A-Za-z]+)")
    text = p.sub('\\1 minion', text)
    return text


def create_vocabulary(text, token_len=None, bgr_len=None):
    # preprocessing
    text = clean_text(text)
    text = replace_common_text(text)

    # tokenize
    tokens = text.split(' ')
    stemmer = PorterStemmer()

    # stem and clean even further
    tokens = [stemmer.stem(w).lower() for w in tokens if
              (w not in stopwords.words('english')) and (w not in string.punctuation)]

    # get most common tokens
    fdist = FreqDist(tokens)
    vocabulary = [t[0] for t in fdist.most_common(token_len)]

    # get most common bigrams
    bigrams = nltk.bigrams(tokens)
    fdist = FreqDist(bigrams)
    vocabulary.extend(["{} {}".format(*bigram_tuple[0]) for bigram_tuple in fdist.most_common(bgr_len)])

    return vocabulary


def create_description_vector(card, vocab):
    x = np.zeros(len(vocab))

    card_vocab = create_vocabulary(card.description)

    for token in card_vocab:
        try:
            x[vocab.index(token)] = 1.
        except ValueError:
            continue

    return x


def minion_tensor(card, vocab):
    if card.type is not CardType.MINION:
        raise ValueError("Must be a minion card")

    x = np.array([
        card.atk,
        card.health,
        card.divine_shield,
        GameTag.CHARGE in card.tags,
        GameTag.STEALTH in card.tags,
        card.windfury,
        card.taunt,
        card.spell_damage,
        card.overload,
        card.deathrattle,
        card.poisonous,

        GameTag.BATTLECRY in card.tags,
        GameTag.CANT_BE_TARGETED_BY_HERO_POWERS in card.tags,
        GameTag.CANT_ATTACK in card.tags,

    ])
    x = x.astype(np.float32)

    if vocab is not None:
        x = np.concatenate([x, create_description_vector(card, vocab)])

    return x


def batch(iterable, n=10):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]




def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def get_tensors(minions, vocab):
    costs_tensors = [np.array([i == x.cost for i in range(MANA_COST_RANGE)]).astype(np.float32) for x in minions]
    minion_tensors = [minion_tensor(c, vocab) for c in minions]
    return np.array(minion_tensors), np.array(costs_tensors)


def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer



def create_graph(vocab):


    minions = list(filter(lambda x: x.type == CardType.MINION and x.collectible, db.values()))

    train_size = 0.8
    train_cnt = int(len(minions) * train_size)

    train_minions = minions[:train_cnt]
    test_minions = minions[train_cnt:]

    x_train, y_train = get_tensors(train_minions, vocab)

    x_test, y_test = get_tensors(test_minions, vocab)


    n_input = x_train.shape[1]
    n_classes = y_train.shape[1]

    n_hidden_1 = (n_input + n_classes) // 2

    print(n_input)
    print(n_classes)
    print(n_hidden_1)

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    keep_prob = tf.placeholder("float")

    training_epochs = 5000
    display_step = 1000
    batch_size = 32

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    predictions = multilayer_perceptron(x, weights, biases, keep_prob)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(x_train) / batch_size)
            x_batches = np.array_split(x_train, total_batch)
            y_batches = np.array_split(y_train, total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: 0.8
                                })
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: x_test, y: y_test, keep_prob: 1.0}))

        for minion in test_minions:
            print(minion)
            input_minion, input_cost = get_tensors([minion], vocab)

            print(minion.description)
            #print(minion_tensor(minion, vocab))
            #print(sess.run(predictions, feed_dict={x: input_minion, keep_prob: 1.0}))
            print(sess.run(tf.argmax(predictions, 1), feed_dict={x: input_minion, keep_prob: 1.0}))
            print(minion.cost)

            input()


def search_card():
    name = input("Enter a card name: ")

    for card in db.values():

        if card.type is not CardType.MINION:
            continue

        if name.lower() in card.name.lower():
            print(card.name)

            description = cleanhtml(card.description)

            print(card.description)
            print(card.type)
            print(card.tags)
            print(card.atk)
            print(card.health)


if __name__ == '__main__':

    if not os.path.isfile(metadecks_path):
        download_decks(metastats_url, metadecks_path)
    elif datetime.fromtimestamp(creation_date(metadecks_path)) < datetime.now() - timedelta(days=3):
        os.remove(metadecks_path)
        download_decks(metastats_url, metadecks_path)

    if not os.path.isfile(carddefs_path):
        download_carddefs(carddefs_url, carddefs_path)
    elif datetime.fromtimestamp(creation_date(carddefs_path)) < datetime.now() - timedelta(days=3):
        os.remove(carddefs_path)
        download_carddefs(carddefs_url, carddefs_path)

    print("Loading CardDefs.xml ...", end='')
    db, _ = hearthstone.cardxml.load(carddefs_path)

    tree = ET.parse(metadecks_path)
    root = tree.getroot()

    print('Done.')

    decks = []

    print('Parsing Decks...', end='')
    for deck in root:
        d = Deck(deck.find('DeckId').text, deck.find('Class').text)

        for card in deck.find('Cards'):
            for i in range(int(card.find('Count').text)):
                d.cards.append(db[card.find('Id').text])

        decks.append(d)
    print('Done.')

    text = ""
    for i in db.values():
        if i.type == CardType.MINION and i.collectible:
            text += i.description.replace('\n', ' ') + '\n'

    vocab = create_vocabulary(text, 120, 50)

    #while True:
    #    search_card()
    create_graph(vocab)

