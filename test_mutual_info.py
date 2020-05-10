import pandas as pd
from pytest import approx

from mutual_info import calc_mutual_information, calc_conditional_entropy_x, calc_mutual_information_for_word

# MI expected: 0.2141709
data1 = [
[0,	0,	0.1],
[0,	1,	0.7],
[1,	0,	0.15],
[1,	1,	0.05]]

# MI expected: 0.375
data2 = [
['A', 'Very low', 1/8],
['B', 'Very low', 1/16],
['AB', 'Very low', 1/32],
['O', 'Very low', 1/32],

['A', 'Low', 1/16],
['B', 'Low', 1/8],
['AB', 'Low', 1/32],
['O', 'Low', 1/32],

['A', 'Medium', 1/16],
['B', 'Medium', 1/16],
['AB', 'Medium', 1/16],
['O', 'Medium', 1/16],

['A', 'High', 1/4],
['B', 'High', 0],
['AB', 'High', 0],
['O', 'High', 0],
]

# MI expected:  0.1076398503
data3 = [
    ['paint', 'art', 12/102],
    ['not paint', 'art', 45/102],
    ['paint', 'music', 0],
    ['not paint', 'music', 45/102]
]

# MI expected: 0.214170945007629
data4 = [
    ['paint', 'art', 0.1],
    ['paint', 'music', 0.7],
    ['not paint', 'art', 0.15],
    ['not paint', 'music', 0.05]
]

data_weather = [
    ['calm', 'N',  0.00],
    ['normal',  'N', 0.25],
    ['windy',  'N',  0.75],
    ['calm', 'Y',  0.3636],
    ['normal',  'Y',  0.4545],
    ['windy',  'Y',  0.1818]
]

# def test_conditional_entropy():
#    df = pd.DataFrame(data3, columns=['x', 'y', 'prob'])
#    h_c_paint = calc_conditional_entropy_x(df,  "paint")
#    assert h_c_paint == 0

#h_c_not_paint = calc_conditional_entropy_x(df,  "not paint")
#    assert h_c_not_paint == 1


def test_calc_mutual_information_for_word():
    test_data = {0: [data4, 0.2141709], 1: [data4, 0.214170945007629], 2: [data2, 0.375], 3: [data3, 0.1076398503] }

    for k in test_data:
        df = pd.DataFrame(test_data[k][0], columns=['x', 'y', 'prob'])
        mi_ce = calc_mutual_information_for_word(df, use_cond_entropy=True)
        mi_je = calc_mutual_information_for_word(df, use_cond_entropy=False)
        print(f"Mutual information mi_ce: {mi_ce}  Mutual information mi_je: {mi_je}")
        assert mi_ce == approx(mi_je, rel=1e-5)
        assert mi_ce == approx(test_data[k][1], rel=1e-5)


def test_mutual_information():
    data3b = [
        ['paint', 'art', 12],
        ['paint', 'music', 0],
    ]

    data3b_class_counts = {'art': 57, 'music': 45}
    test_result = [ 0.1076398503]

    df = pd.DataFrame(data3b, columns=['x', 'y', 'prob'])
    mi_ce = calc_mutual_information(df, ['paint'], data3b_class_counts, use_cond_entropy=True)
    print(f"Mutual info: {mi_ce}")
    assert mi_ce[0] == approx(test_result[0], rel=1e-5)