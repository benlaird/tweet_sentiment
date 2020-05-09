from math import log2

import numpy as np
import pandas as pd


def calc_pmi(series):
    global px, py

    prob = series['prob']
    if prob == 0:
        prob = np.nextafter(0, 1)  # Smallest positive number after zero towards 1
    pmi = log2(prob) - log2(px[series['x']] * py[series['y']])
    return pmi


def calc_mi(series):
    mi = calc_pmi(series) * series['prob']
    return mi


def calc_joint_entropy(series):
    prob = series['prob']
    if prob == 0:
        return 0
    je = -1 * prob * log2(prob)
    return je


def calc_entropy(marginals : dict):
    """
    Given a dict of marginal probabilities calculate entropy
    :param marginals:
    :return:
    """
    h = 0
    for p in marginals:
        if marginals[p] == 0:
            continue
        h += marginals[p] * log2(marginals[p])
    h *= -1
    return h


def calc_conditional_entropy_x(df, x_value):
    vec_df = df[df['x'] == x_value].copy()
    vec_df['prob'] = vec_df['prob'] / vec_df['prob'].sum()

    h = 0
    for prob in vec_df['prob']:
        if prob != 0:
            h += prob * log2(prob)
    if h != 0:
        h *= -1
    return h


def calc_conditional_entropy_over_all_x(df, px):
    h = 0
    gs = df.groupby('x').groups
    for x in gs:
        h += px[x] * calc_conditional_entropy_x(df, x)
    return h


def calc_mutual_information_using_joint_entropy(df, hx, hy):
    df['je'] = df.apply(calc_joint_entropy, axis=1)
    je = df['je'].sum()
    mut_info = hx + hy - je
    return mut_info


def calc_mutual_information_using_cond_entropy(df, px, py):
    hy = calc_entropy(py)
    h = calc_conditional_entropy_over_all_x(df, px)
    mi = hy - h
    return mi


def calc_mutual_information(df, use_cond_entropy=True):
    debug = False

    # Calc marginal probabilities for x
    gs = df.groupby('x').groups
    px = {}
    for g in gs:
        px[g] = df.iloc[gs[g]].sum()['prob']

    # Calc marginal probabilities for y
    gs = df.groupby('y').groups
    py = {}
    for g in gs:
        py[g] = df.iloc[gs[g]].sum()['prob']

    hx = calc_entropy(px)
    hy = calc_entropy(py)

    if debug:
        print(f"px marginals: {px}")
        print(f"py marginals: {py}")
        print(df)
        print(f"H(X): {hx}  H(Y): {hy} ")

    mi = 0
    if use_cond_entropy:
        mi = calc_mutual_information_using_cond_entropy(df, px, py)
    else:
        mi = calc_mutual_information_using_joint_entropy(df, hx, hy)

    return mi

