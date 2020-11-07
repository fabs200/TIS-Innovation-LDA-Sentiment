import pandas
from python.params import params as p

def filter_sentiment_params(df, df_sentiment_list):
    """
    filters out Sentiment params specified in params.py, this filter is used only in 06*_LDA_Plots_*.py
    Filters: 'drop_sentence_lenght', 'drop_article_lenght', 'drop_prob_below', 'drop_senti_below', 'drop_senti_above'
    :param df: lda_results_*_l.csv
    :param df_sentiment_list: 'sepldefault', 'seplmodified', 'sentiwsdefault', 'sentifinal'
    :return: filtered pandas df
    """

    # drop short articles
    if p['drop_article_lenght']:
        df['articles_text_lenght'] = df['articles_text'].str.len()
        df = df.drop(df[df.articles_text_lenght <= p['drop_article_lenght']].index)

    # drop short sentences
    if p['drop_sentence_lenght']:
        df['sentences_for_sentiment_lenght']= df['sentences_for_sentiment'].str.len()
        df = df.drop(df[df.sentences_for_sentiment_lenght <= p['drop_sentence_lenght']].index)

    # drop articles with low probability of assigned dominant topic
    if p['drop_prob_below']:
        df['DomTopic_arti_arti_prob'] = pandas.to_numeric(df['DomTopic_arti_arti_prob'])
        df = df.drop(df[df.DomTopic_arti_arti_prob <= p['drop_prob_below']].index)

        # set main sentiscore_mean, rename and to numeric
        df['sentiscore_mean'] = pandas.to_numeric(df['sentiscore_mean'], errors='coerce')

    # drop sentences with (relatively) neutral sentiment score (either =0 or in range(-.1, .1)
    if p['drop_senti_below']:
        df = df.drop(df[(df.sentiscore_mean <= p['drop_senti_below']) &
                        (df.sentiscore_mean >= p['drop_senti_above'])].index)

    return df


