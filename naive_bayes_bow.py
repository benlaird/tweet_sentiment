import re

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from yellowbrick.classifier import classification_report as yb_class_report, ClassificationReport
from yellowbrick.classifier import ConfusionMatrix

import spacy
from spacy.matcher.phrasematcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")


class Debug(TransformerMixin, BaseEstimator):

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        self.fit_result = []
        self.transform_result = []

    def fit(self, X, y=None):
        # Return the transformer
        self.fit_result = X
        return self

    def transform(self, X):
        # No op transform
        self.transform_result = X
        return X



pipe = Pipeline([
    ("tf_idf", TfidfVectorizer()),
    ("debug", Debug()),
    ("nmf", NMF())
])



def top_n_features(feature_names, response, top_n=3):
    sorted_nzs = np.argsort(response.data)[:-(top_n + 1):-1]
    print(feature_names[response.indices[sorted_nzs]])


def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    """ Top tfidf features in specific document (matrix row) """
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    """ Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. """
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)


def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df


def count_zero_categories(X_2_3_4_docs):
    count_vectorizer = CountVectorizer(analyzer='word',
                                       strip_accents='unicode',
                                       stop_words='english',
                                       lowercase=True,
                                       # token_pattern=r'\b[a-zA-Z]{3,}\b',
                                       # max_df= 1.0,
                                       # min_df = 0.2
                                       )
    X_cat_2_3_4_counts = count_vectorizer.fit_transform(X_2_3_4_docs)
    vectorizer = TfidfTransformer()
    dtm_tfidf_zero_cat = vectorizer.fit_transform(X_cat_2_3_4_counts)
    feature_names = count_vectorizer.get_feature_names()
    print("*** Top mean zero-cat features ***")
    top_mean_f = top_mean_feats(dtm_tfidf_zero_cat, feature_names)
    print(top_mean_f)
    return dtm_tfidf_zero_cat


def all_tweet_text(csv_file, max_rows=-1):
    # keep_default_na=False means empty strings are kept as is, i.e. as ''
    df = pd.read_csv(csv_file, keep_default_na=False)
    # Replace any empty text with empty string instead of np.nan
    # df['text'] = df['text'].replace(np.nan, '', regex=True)
    tweet_text = df['text'].tolist()
    sentiment = df['sentiment'].tolist()

    if max_rows > 0:
        tweet_text = tweet_text[0:max_rows]
        sentiment = sentiment[0:max_rows]
    # new_list = ['+' if int(el)>10 else '-' for el in li]
    # return tweet_text[310:316], sentiment[310:316]
    return tweet_text, sentiment


def sentiment_matches(idx, y, sentiment):
    return y[idx] == sentiment


def custom_tokenizer(sentence):
    """
        remove stop words, and lemmatize
    """
    tokens = nlp(sentence)
    ret = [token.lemma_ for token in tokens if not (token.is_punct | token.is_space | token.is_stop)]
    return ret


def model_naive_bayes():
    use_counts = False
    debug = True

    # Read all tweet text into docs
    # Docs is an array of text strings
    if debug:
        docs, dependent_var = all_tweet_text("./data/tweet-sentiment-extraction/train.csv", max_rows=100)
    else:
        docs, dependent_var = all_tweet_text("./data/tweet-sentiment-extraction/train.csv")

    X = docs
    y = dependent_var

    indices = range(0, len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, random_state=1,
                                                                             test_size=0.25)

    vectorizer = TfidfVectorizer(analyzer='word',
                                 strip_accents='unicode',
                                 stop_words='english',
                                 lowercase=True,
                                 tokenizer=custom_tokenizer,
                                 ngram_range=(1, 2)
                                 # token_pattern=r'\b[a-zA-Z]{3,}\b',
                                 # max_df=0.6,
                                 # min_df=0.0
                                 )
    dtm_tfidf_train = vectorizer.fit_transform(X_train)

    feature_names = np.array(vectorizer.get_feature_names())

    # Return the idx's of the positive sentiments
    positive_train = list(filter(lambda idx: sentiment_matches(idx, y, 'positive'), idx_train))
    positive_train_offset = []
    for i in range(0, dtm_tfidf_train.get_shape()[0]):
        if idx_train[i] in positive_train:
            positive_train_offset.append(i)

    # Return the idx's of the negative sentiments
    negative_train = list(filter(lambda idx: sentiment_matches(idx, y, 'negative'), idx_train))
    negative_train_offset = []
    for i in range(0, dtm_tfidf_train.get_shape()[0]):
        if idx_train[i] in negative_train:
            negative_train_offset.append(i)

    if False:
        # for i in range(0, dtm_tfidf_train.get_shape()[0]):
        for i in range(0, dtm_tfidf_train.get_shape()[0]):
            if idx_train[i] in positive_train:
                response = dtm_tfidf_train[i]
                tweet = X[idx_train[i]]
                print(f"Id: {i} Article: {tweet}")
                top_n_features(feature_names, response, 5)
                print("\n\n")

    print(f"Train type: {type(dtm_tfidf_train)} Train shape: {dtm_tfidf_train.shape}")

    dtm_tfidf_test = vectorizer.transform(X_test)

    if False:
        for response in dtm_tfidf_test:
            top_n_features(feature_names, response)

    gnb = MultinomialNB()
    gnb.fit(dtm_tfidf_train, y_train)

    nb_train_preds = gnb.predict(dtm_tfidf_train)
    nb_test_preds = gnb.predict(dtm_tfidf_test)

    nb_train_score = accuracy_score(y_train, nb_train_preds)
    nb_test_score = accuracy_score(y_test, nb_test_preds)

    print("Multinomial Naive Bayes")
    print("Training Accuracy: {:.4} \t\t Testing Accuracy: {:.4}".format(nb_train_score, nb_test_score))
    print("")
    print('-' * 70)
    print("")

    # Confusion matrix and classification report
    c_m = confusion_matrix(y_test, nb_test_preds)
    print(c_m)
    print(classification_report(y_test, nb_test_preds))

    print("*** Top mean positive features ***")
    # top_mean_f = top_mean_feats(dtm_tfidf_train, feature_names)
    top_mean_f = top_mean_feats(dtm_tfidf_train, feature_names, grp_ids=positive_train_offset)
    print(top_mean_f)

    print("*** Top mean negative features ***")
    # top_mean_f = top_mean_feats(dtm_tfidf_train, feature_names)
    top_mean_f = top_mean_feats(dtm_tfidf_train, feature_names, grp_ids=negative_train_offset)
    print(top_mean_f)

    if True:
        visualizer = ClassificationReport(gnb)  # classes=classes, support=True)
        visualizer.fit(dtm_tfidf_train, y_train)
        visualizer.score(dtm_tfidf_test, y_test)  # Evaluate the model on the test data
        visualizer.show()  # Finalize and show the figure


def main():
    model_naive_bayes()


main()
