# naive_bayes_bow.py

import os
import pickle
import re
import time

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline


from yellowbrick.classifier import classification_report as yb_class_report, ClassificationReport
from yellowbrick.classifier import ConfusionMatrix

import spacy
from spacy.matcher.phrasematcher import PhraseMatcher

if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE', None):
    from analyze_cond_probs import read_cond_probs
    from feature_selection import mutual_info_best
    from globals import Global

Global()
nlp = None

class Debug(TransformerMixin, BaseEstimator):
    """
    This class is used to debug pipelines, it saves the current value of X from both the previous pipeline step
    that could have been a fit or a transform
    """

    def __init__(self):
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


class ClockTime:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def start(self, info_str=None):
        if info_str:
            print(info_str)
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed

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


def all_tweet_text(csv_file, max_rows=None):
    # keep_default_na=False means empty strings are kept as is, i.e. as ''
    df = pd.read_csv(csv_file, keep_default_na=False)
    # Replace any empty text with empty string instead of np.nan
    # df['text'] = df['text'].replace(np.nan, '', regex=True)
    tweet_text = df['text'].tolist()
    sentiment = df['sentiment'].tolist()
    textID = df['textID'].tolist()

    if max_rows:
        tweet_text = tweet_text[0:max_rows]
        sentiment = sentiment[0:max_rows]
        textID  = textID[0:max_rows]
    # new_list = ['+' if int(el)>10 else '-' for el in li]
    # return tweet_text[310:316], sentiment[310:316]
    return tweet_text, sentiment, textID


def sentiment_matches(idx, y, sentiment):
    return y[idx] == sentiment


def custom_tokenizer(sentence):
    """
        remove stop words, and lemmatize
    """
    tokens = nlp(sentence)
    ret = [token.lemma_ for token in tokens if not (token.is_punct | token.is_space | token.is_stop)]
    return ret


def top_feature_in_sentence(sentence, feat_cond_probs):
    tokens = nlp(sentence)

    d = {}
    # Make a dictionary of all the tokens with the value as the original text
    i = 0
    for i in range(0, len(tokens)):
        token = tokens[i]
        if not (token.is_punct | token.is_space | token.is_stop):
            d[token.lemma_] = (i, token)

    ret_str = ""
    for feature in feat_cond_probs:
        if feature in d:
            i, token = d[feature]
            res_str = token.text
            if i+1 < len(tokens) and tokens[i+1].is_punct:
                res_str = res_str + tokens[i+1].text
            return res_str

    print(f"No feature found for {sentence}")
    return None



def create_pipeline(read_vector_cache, vectorizer_pipe_name, k_best):

    if read_vector_cache:
        with open('vectorizer.pickle', 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            vectorizer = pickle.load(f)
            vect_pipe_tupe = (vectorizer_pipe_name, vectorizer)
    else:
        vect_pipe_tupe = (vectorizer_pipe_name, CountVectorizer(analyzer='word',
                                                                strip_accents='unicode',
                                                                stop_words='english',
                                                                lowercase=True,
                                                                tokenizer=custom_tokenizer,
                                                                # ngram_range=(1, 2)
                                                                ))

    pipe = Pipeline([
        vect_pipe_tupe,
        ("tf_idf_debug", Debug()),
        ('best_mutual_info', SelectKBest(mutual_info_best, k=k_best)),
        ('kbest_debug', Debug()),
        ('clf', MultinomialNB())])

    Global.set_pipe(pipe)
    return pipe

def split_data(debug_max_rows=None, real_test=False):
    """

    :param real_test: if true load the test file as the test data
    :return:  X_train, X_test, y_train, y_test, idx_train, idx_test, textID_test
    """
    # If not a real test split the training data into train test portions
    if not real_test:

        # Read all tweet text into docs
        # Docs is an array of text strings

        docs, dependent_var, train_textID = all_tweet_text(Global.input_dir + "train.csv", max_rows=debug_max_rows)

        X = docs
        y = dependent_var

        indices = range(0, len(X))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, random_state=1,
                                                                                 test_size=0.25)
    else:
        X_train, y_train, train_textID = all_tweet_text(Global.input_dir + "train.csv", max_rows=debug_max_rows)
        idx_train = range(0, len(X_train))

        X_test, y_test, test_textID = all_tweet_text(Global.input_dir + "test.csv", max_rows=debug_max_rows)
        idx_test = range(0, len(X_test))

    return X_train, X_test, y_train, y_test, idx_train, idx_test, test_textID



def model_naive_bayes(k_best=100, real_test=False, save_vector_cache=False, read_vector_cache=False, debug_max_rows=None):
    use_counts = False
    ct = ClockTime()

    vectorizer_pipe_name = "count"

    X_train, X_test, y_train, y_test, idx_train, idx_test, test_textID = split_data(debug_max_rows, real_test)

    pipe = create_pipeline(read_vector_cache, vectorizer_pipe_name, k_best)

    ct.start("Fitting the training data...")
    result = pipe.fit(X_train, y_train)
    print(f"Fitted the training data in {ct.stop()} seconds")

    vectorizer = pipe[vectorizer_pipe_name]
    if save_vector_cache:
        with open('vectorizer.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(vectorizer, f, pickle.HIGHEST_PROTOCOL)

    feature_names = np.array(vectorizer.get_feature_names())

    dtm_tfidf_train = pipe['tf_idf_debug'].fit_result

    # Return the idx's of the positive sentiments
    positive_train = list(filter(lambda idx: sentiment_matches(idx, y_train, 'positive'), idx_train))
    positive_train_offset = []
    for i in range(0, dtm_tfidf_train.get_shape()[0]):
        if idx_train[i] in positive_train:
            positive_train_offset.append(i)

    # Return the idx's of the negative sentiments
    negative_train = list(filter(lambda idx: sentiment_matches(idx, y_train, 'negative'), idx_train))
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

    dtm_tfidf_train = pipe['tf_idf_debug'].fit_result

    print(f"Train shape: {dtm_tfidf_train.shape}")

    ct.start("Predicting the training & test data...")
    nb_train_preds = pipe.predict(X_train)
    nb_test_preds = pipe.predict(X_test)
    print(f"Did the train & test predictions in: {ct.stop()} seconds")

    nb_train_score = accuracy_score(y_train, nb_train_preds)
    nb_test_score = accuracy_score(y_test, nb_test_preds)

    print("Multinomial Naive Bayes")
    print(f"*** K-best == {k_best}")
    print("Training Accuracy: {:.4} \t\t Testing Accuracy: {:.4}".format(nb_train_score, nb_test_score))
    print("")
    print('-' * 70)
    print("")

    """    print("*** Top mean positive features ***")
        top_mean_f = top_mean_feats(dtm_tfidf_train, feature_names, grp_ids=positive_train_offset)
        print(top_mean_f)
    
        print("*** Top mean negative features ***")
        top_mean_f = top_mean_feats(dtm_tfidf_train, feature_names, grp_ids=negative_train_offset)
        print(top_mean_f)"""

    # Confusion matrix and classification report
    c_m = confusion_matrix(y_test, nb_test_preds)
    print(c_m)
    print(classification_report(y_test, nb_test_preds))

    cond_probs_file_name = "cond_probs.json"
    feat_cond_probs = read_cond_probs(cond_probs_file_name)
    # print(cond_probs)

    i = 0
    selected_text = []
    for i in range(0, len(X_test)):
        sentence = X_test[i]
        textID = test_textID[i]
        if y_test[i] == 'neutral':
            top_feat = None
        else:
            top_feat = top_feature_in_sentence(sentence, feat_cond_probs)
        if top_feat:
            selected_text.append(top_feat)
        else:
            selected_text.append(sentence)
        print(f"textID: {textID} Sentence: {sentence} top_feat: {top_feat}")

    data = {'textID' : test_textID, 'selected_text' : selected_text}
    submission_df = pd.DataFrame(data)
    submission_df.set_index('textID', inplace=True)
    submission_df.to_csv("submission.csv", sep=',')


    if False:
        gnb = pipe['clf']
        # Call just the transform on the test data - so we can visualize the fit later!
        dtm_tfidf_test = pipe['tf_idf'].transform(X_test)
        # for response in dtm_tfidf_test:
        #    top_n_features(feature_names, response)

        visualizer = ClassificationReport(gnb)  # classes=classes, support=True)
        visualizer.fit(dtm_tfidf_train, y_train)
        visualizer.score(dtm_tfidf_test, y_test)  # Evaluate the model on the test data
        visualizer.show()  # Finalize and show the figure

def cross_validate(k_best=100, read_vector_cache=False, debug=False, if_debug_max_rows=1000):
    vectorizer_pipe_name = "count"

    # Read all tweet text into docs
    # Docs is an array of text strings
    if debug:
        docs, dependent_var = all_tweet_text("./data/tweet-sentiment-extraction/train.csv", max_rows=if_debug_max_rows)
    else:
        docs, dependent_var = all_tweet_text("./data/tweet-sentiment-extraction/train.csv")

    X = docs
    y = dependent_var

    indices = range(0, len(X))
    pipe = create_pipeline(read_vector_cache, vectorizer_pipe_name, k_best)
    vect_pipe_tupe = (vectorizer_pipe_name, CountVectorizer(analyzer='word',
                                           strip_accents='unicode',
                                           stop_words='english',
                                           lowercase=True,
                                           tokenizer=custom_tokenizer,
                                           # ngram_range=(1, 2)
                                           ))
    pipe = Pipeline([
        vect_pipe_tupe,
        # ("tf_idf_debug", Debug()),
        # ('best_mutual_info', SelectKBest(mutual_info_best, k=k_best)),
        # ('kbest_debug', Debug()),
        ('clf', MultinomialNB())])

    cv =  StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_micro", n_jobs = 1)
    mean_score = scores.mean()
    print(f"Scores: {scores}")
    print(f"Mean f1-micro score: {mean_score}")


def main():
    global nlp
    nlp = spacy.load("en_core_web_sm")

    slang_typo_normalization = {
        "cos": "because",
        "fav": "favorite",
        'awsome': 'awesome',
        'thanx': 'thanks',
        'luv': 'love',
        'mom' : 'mother',
        'mommy' : 'mother',
        'congrats' : 'congratulations',
        'congrat' : 'congratulation'
    }

    # Add special cases for slang & typos
    for k in slang_typo_normalization:
        nlp.tokenizer.add_special_case(k, [{'ORTH': slang_typo_normalization[k]}])

    # model_naive_bayes(save_vector_cache=True)

    # model_naive_bayes(debug_max_rows=5000)
    # cross_validate(k_best=100)

    model_naive_bayes(real_test=True, debug_max_rows=1000, k_best=10)
    # model_naive_bayes(k_best=150)
main()
