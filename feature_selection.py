import json

from IPython.display import display, HTML
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

from mutual_info import calc_mutual_information,  conditional_probability_for_y_given_partial_jpd

from globals import Global

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


def sparse_to_df(sp_matrix, feat_names):
    return pd.DataFrame(sp_matrix.todense(), columns=feat_names)


def convert_log_prob_to_df_prob(log_prob, class_names, k_best_feat):
    """
    Converts the log probabilities returned by MultinomialNB to a data frame

    :param log_prob:
    :param class_names:
    :return:
    """
    # Transpose the class labels into the columns
    prob_arr = np.array(log_prob)
    prob_arr = np.transpose(prob_arr)

    probs = pd.DataFrame(prob_arr, columns=class_names)

    # Unlog by taking exponents
    probs = probs.apply(np.exp)
    # Add the feature names as the first column
    probs['feature'] = k_best_feat

    # Unpivot the class names from the columns to the rows.. Note this doesn't work if the index is already set
    probs = pd.melt(probs, id_vars=['feature'], value_vars=class_names, var_name='y', value_name='prob')
    return probs


def compute_relative_freq_df_old(df, feat_names, class_names, actuals_class_name, compute_relative_prob=True):
    # For each feature, calculate it's p(feature | outcome)
    # puts each separate combination of feature & outcome in a separate row
    debug = False
    rows = []

    for f in feat_names:
        for c in class_names:
            # Each row is the feature name, class followed by the value_counts of the 1's
            vcs = [f, c]

            vc = df[df[actuals_class_name] == c][f].value_counts()
             # Get the 1's
            vc = vc.get(1) if vc.get(1) else 0
            vcs.append(vc)
            if debug:
                print(f"f: {f} c: {c} vc: {vc}")
            rows.append(vcs)

    cols = ['x', actuals_class_name, 'prob']
    # cols.extend(class_names)

    likelihoods_df = pd.DataFrame(rows, columns=cols)

    if compute_relative_prob:
        for c in class_names:
            prob_sum = likelihoods_df.loc[likelihoods_df[actuals_class_name] == c, 'probability'].sum()
            likelihoods_df.loc[likelihoods_df[actuals_class_name] == c, ['probability']] = \
                likelihoods_df.loc[likelihoods_df[actuals_class_name] == c, ['probability']] / prob_sum

    return likelihoods_df


def compute_relative_freq_df(df, feat_names, class_names, actuals_class_name, compute_relative_prob=True):
    # For each feature, calculate it's p(feature | outcome)
    # puts each separate combination of feature & outcome in a separate row
    debug = False
    rows = []

    # Sum the document counts by class label
    gs = df.groupby('y').sum()

    # Transpose and rename the index col
    gs2 = gs.transpose().rename_axis('feature', axis=1)

    # melt seems to require there to be no index
    gs2.reset_index(inplace=True)

    # Make the distinct class names distinct rows
    likelihoods_df = pd.melt(gs2, id_vars=['index'], value_vars=class_names, var_name='y')

    cols = ['x', actuals_class_name, 'prob']
    likelihoods_df.columns = cols

    if compute_relative_prob:
        for c in class_names:
            prob_sum = likelihoods_df.loc[likelihoods_df[actuals_class_name] == c, 'probability'].sum()
            likelihoods_df.loc[likelihoods_df[actuals_class_name] == c, ['probability']] = \
                likelihoods_df.loc[likelihoods_df[actuals_class_name] == c, ['probability']] / prob_sum

    return likelihoods_df

weather = [
    ['Sunny', 'Hot', 'Normal', 'Calm', 'Y'],
    ['Overcast', 'Mild', 'Normal', 'Calm', 'Y'],
    ['Sunny', 'Cool', 'Normal', 'Windy', 'Y'],
    ['Sunny', 'Hot', 'Normal', 'Windy', 'N'],
    ['Overcast', 'Cool', 'Humid', 'Windy', 'N'],
    ['Sunny', 'Mild', 'Humid', 'Calm', 'Y'],
    ['Overcast', 'Mild', 'Normal', 'Calm', 'Y'],
    ['Rainy', 'Cool', 'Humid', 'Windy', 'N'],
    ['Rainy', 'Hot', 'Normal', 'Windy', 'Y']]


def likelihood_best(X, y):
    """
    Custom function to be used by SelectKBest that computes the likelihood for each feature, class combination
    I.e. P(feature | class)

    Currently only handles the two class case.. For other cases, will return an array of zeros
    :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Sample vectors.
    :param y: array-like of shape (n_samples,)
        Target vector (class labels).
    :return: array, shape = (n_features,)
        Absolute likelihood difference vector
    """
    pipe = Global.get_pipe()

    feat_names = pipe['count'].get_feature_names()
    # Initialize return array to all zeroes
    a1 = np.array([0] * len(feat_names))

    count_vec_df = sparse_to_df(X, feat_names)
    count_vec_df['y'] = y
    cls_names = np.unique(y)
    likelihoods_df = compute_relative_freq_df(count_vec_df, feat_names,
                                              cls_names, 'Actuals')
    # Two class case
    if len(cls_names) == 2:
        a1 = []
        for f in feat_names:
            diff = abs(likelihoods_df[likelihoods_df['features'] == f].iloc[0]['probability'] - \
                    likelihoods_df[likelihoods_df['features'] == f].iloc[1]['probability'] )

            a1.append(diff)
        a1 = np.array(a1)
    print(f"a1: {a1}")
    return a1

def mutual_info_best(X, y):
    """
    Custom function to be used by SelectKBest that computes the mutual information for each feature, class combination
    I.e. P(feature | class)

    Currently only handles the two class case.. For other cases, will return an array of zeros
    :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Sample vectors.
    :param y: array-like of shape (n_samples,)
        Target vector (class labels).
    :return: array, shape = (n_features,)
        Absolute likelihood difference vector
    """
    debug=True
    pipe = Global.get_pipe()

    feat_names = pipe['count'].get_feature_names()
    # Initialize return array to all zeroes
    a1 = np.array([0] * len(feat_names))

    count_vec_df = sparse_to_df(X, feat_names)
    count_vec_df['y'] = y
    cls_names = np.unique(y)

    cls_counts = {}
    gs = count_vec_df.groupby('y').groups
    for y in gs:
        cls_counts[y] = len(gs[y])
    print(f"Class count residuals: {cls_counts}")

    likelihoods_df = compute_relative_freq_df(count_vec_df, feat_names,
                                              cls_names, 'y', compute_relative_prob=False)

    mi = calc_mutual_information(likelihoods_df, feat_names, cls_counts, use_cond_entropy=True)
    if debug:
        topN = 150
        print(f"feature names: {feat_names}")
        print(f"mutual info: {mi}")
        # Make a data frame from the feature names & their mutual info
        s1 = pd.Series(feat_names, name='features')
        s2 = pd.Series(mi,  name='mi')
        f_df = pd.concat([s1, s2], axis=1)
        f_df.set_index('features', inplace=True)
        f_df = f_df.sort_values(by=['mi'], ascending=False)
        print(f_df)
        f_df.head(topN).to_csv("mi_feats.tsv", sep='\t')

        topn_feat_names = list(f_df.index[:topN])
        cond_probs = conditional_probability_for_y_given_partial_jpd(likelihoods_df, topn_feat_names)
        with open('cond_probs.json', 'w') as fp:
            json.dump(cond_probs, fp, indent=4)

        print(f"cond_probs: {cond_probs}")
    mi = np.array(mi)
    return mi


def test_feature_selection():
    # Create the pandas DataFrame
    df = pd.DataFrame(weather, columns=['Weather', 'Temperature', 'Humidity', 'Wind', 'y'])
    df_text = df["Weather"] + " " + df["Temperature"] + " " + df["Humidity"] + " " + df["Wind"]

    X = df_text
    y = df['y']
    k_best = 3

    display(df)
    # Pipeline, note that alpha is set to zero in MultinomialNB just to validate the likelihood computations
    pipe = Pipeline([('count', CountVectorizer()),
                     ('countvectorizer_debug', Debug()),
                     # ('tf_idf', TfidfTransformer(norm=None)),
                     # ('tf_idf_debug', Debug()),
                     # ('chi2', SelectKBest(chi2, k=k_best)),
                     # ('best_likelihoods', SelectKBest(likelihood_best, k=k_best)),
                     ('best_likelihoods', SelectKBest(mutual_info_best, k=k_best)),

                     ('kbest_debug', Debug()),
                     ('clf', MultinomialNB(alpha=0))])

    Global.set_pipe(pipe)
    result = pipe.fit(X, y)

    feat_names = pipe['count'].get_feature_names()

    print("Count vectorizer debug")
    count_vec_df = sparse_to_df(pipe['countvectorizer_debug'].fit_result, pipe['count'].get_feature_names())
    print(count_vec_df)

    print(f"K={k_best} best features accuracy")
    k_best_feat = [feat_names[i] for i in pipe['best_likelihoods'].get_support(indices=True)]
    print(f"k_best_feat: {k_best_feat}")
    k_best_df = sparse_to_df(pipe['kbest_debug'].fit_result, k_best_feat)
    k_best_df['Suitable actuals'] = y

    y_preds = pipe.predict(X)

    # Confusion matrix and classification report
    c_m = confusion_matrix(y, y_preds)
    print(c_m)
    print(classification_report(y, y_preds))

    k_best_df['Suitable predictions'] = pd.Series(y_preds)
    print(f"K={k_best} best features with predictions")
    print(k_best_df)

    probs = convert_log_prob_to_df_prob(pipe['clf'].feature_log_prob_, pipe['clf'].classes_, k_best_feat)
    # Round the columns to account for alpha
    tmp = probs.select_dtypes(include=[np.number])
    probs.loc[:, tmp.columns] = np.round(tmp, 4)
    print(probs)

debug = False
if debug:
    test_feature_selection()