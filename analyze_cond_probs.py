import json
import operator
from statistics import pstdev, mean
import pandas as pd
import numpy as np
from tabulate import tabulate
import plotly.express as px
import plotly.graph_objects as go


class SentimentDispersion:
    def __init__(self, feature, sentiment, prob):
        self.feature = feature
        self.sentiment = sentiment
        self.prob = prob
        self.risk_ratio = None
        self.max_outlier = False

    def set_risk_ratio(self, risk_ratio):
        self.risk_ratio = risk_ratio

    def data_as_dict(self):
        d = {}
        d['feature'] = self.feature
        d['sentiment'] = self.sentiment
        d['prob'] = self.prob
        d['risk_ratio'] = self.risk_ratio
        return d

    def __repr__(self):
        s = f"Feature: {self.feature} Sentiment: {self.sentiment} prob:{self.prob} risk ratio:{self.risk_ratio}"
        return s

    def __str__(self):
        return self.__repr__()


def bar_plot(df, sentiment):
    color_palette_list = px.colors.diverging.Armyrose

    if sentiment == 'positive':
        color = color_palette_list[0]
    elif sentiment == 'neutral':
        color = color_palette_list[3]
    else:
        color = color_palette_list[6]

    layout0 = go.Layout(
        legend=dict(xanchor='center', yanchor='top'),
    )
    # Replace infinity with max of the data frame - excluding nans
    df = df.replace([np.inf, np.inf], np.nan)
    df = df.replace(np.nan, np.nanmax(df["risk_ratio"].values))
    print(df)

    # Use textposition='auto' for direct text
    fig = go.Figure(data=[
        # go.Bar(x=df['feature'], y=df['abs_stdev'], name='Z-score', marker_color=px.colors.qualitative.Set2[0]),
        go.Bar(x=df['feature'], y=df['risk_ratio'], name='Risk ratio for top two category values', marker_color=color),
    ], layout=layout0)
    # Change the bar mode
    fig.update_layout(barmode='group', xaxis_tickangle=-45,
                      showlegend=False,
                      yaxis=dict(
                          title='Risk ratio for top two category values',
                          # titlefont_size=16,
                          # tickfont_size=14,
                      ),
                      )
    # Saving loses data!!
    # fig.write_image(f"{sentiment}.png")
    fig.show()


def scatter_plot(df):
    fig = go.Figure(data=go.Scatter(
        x=df['abs_stdev'],
        y=df['rel_stdev'],
        mode='markers'
    ))

    i = 0
    df = df.sort_values(by=['abs_stdev'])
    for index, row in df.iterrows():
        offset_arr = [-1, +1, -2, +2, -3, +3, -4, +4]
        offset = i % len(offset_arr)
        offset_pct = offset_arr[offset] * 20
        fig.add_annotation(
            x=row['abs_stdev'],
            y=row['rel_stdev'],
            ay=offset_pct,
            text=row['feature'])
        i += 1

    fig.update_layout(
        showlegend=False,
        annotations=[
            dict(
                xref="x",
                yref="y",
                showarrow=True,
                # arrowhead=7,
                ax=0,
                # ay=-40
                textangle=-90,

            )
        ]
    )
    fig.show()


def read_cond_probs(filename):
    with open(filename, "r") as read_file:
        data = json.load(read_file)
    return data


def main():
    debug = False
    cond_probs_file_name = "cond_probs.json"
    cond_probs = read_cond_probs(cond_probs_file_name)
    # print(cond_probs)

    maximal_sds = []

    for feature in cond_probs:
        if feature == "love":
            print("got here")

        probs = [cond_probs[feature][s] for s in cond_probs[feature]]
        m = mean(probs)
        s = pstdev(probs)

        if debug:
            print(f"feature: {feature} probs: {probs}")
            print(f"mean: {m} stdev: {s}")

        sds = []
        # For each sentiment create an object
        for sentiment in cond_probs[feature]:
            value = cond_probs[feature][sentiment]
            sd = SentimentDispersion(feature, sentiment, value)
            sds.append(sd)

        # For each sentiment calculate it's distance from it's nearest neighbor in stdevs
        sds = sorted(sds, key=operator.attrgetter("prob"))
        print(sds)

        # Mark the maximal sd -- these are the ones we care about
        top_sds = sds[len(sds) - 1]
        second_sds = sds[len(sds) - 2]

        # Compute the risk ratio for the topmost category
        if second_sds.prob == 0:
            # top_sds.set_risk_ratio(top_sds.prob / 0.01)
            top_sds.set_risk_ratio(np.inf)
        else:
            top_sds.set_risk_ratio(top_sds.prob / second_sds.prob)

        maximal_sds.append(sds[len(sds) - 1])


    for s in maximal_sds:
        print(s)

    # Convert maximal sds to a dataframe
    df_data = []
    for s in maximal_sds:
        df_data.append(s.data_as_dict())

    # Creates DataFrame.
    df = pd.DataFrame(df_data)  # , index =['feature'])

    df = df.sort_values(by=['risk_ratio'], ascending=False)

    for sentiment in ('positive', 'neutral', 'negative'):
        bar_plot(df[df['sentiment']==sentiment], sentiment)

main()