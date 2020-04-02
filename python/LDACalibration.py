from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

# Todo: change LDACoherence to c_v
def LDACoherence(lda_model=lda_model, corpus=corpus_lda, dictionary=dict_lda, texts=docsforlda):

    # we use coherence measure c_v as suggested by Röder et al. 2015, because it has the highest correlation with human interpretability
    lda_model_cm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence="u_mass")
    #lda_model_cm = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    print(lda_model_cm.get_coherence())

    return lda_model_cm.get_coherence()

def LDACalibration(topics_start, topics_limit, topics_step, dataframecolumn, topn, num_words, metric, no_below=0.1, no_above=0.9, alpha='symmetric', eta=None,
                eval_every=10, iterations=50, random_state=None, verbose=False, display_plot=True):

    metric_values = []
    model_list = [] #Todo: delete

    for num_topics in range(topics_start, topics_limit, topics_step):
        lda_results = EstimateLDA(dataframecolumn, no_below, no_above, num_topics, alpha, eta,
                eval_every, iterations, random_state)
        lda_model = lda_results[0]
        docsforlda = lda_results[1]
        dict_lda = lda_results[2]
        corpus_lda = lda_results[3]
        model_list.append(lda_model)

        if metric == 'coherence':
            metric_values.append(LDACoherence(lda_model=lda_model, corpus=corpus_lda, dictionary=dict_lda, texts=docsforlda))
        if metric == 'jaccard':
            metric_values.append(LDAJaccard(topn=topn, lda_model=lda_model))
        if metric == 'hellinger':
            metric_values.append(LDAHellinger(num_words=num_words, lda_model=lda_model, num_topics=None, dict_lda=dict_lda))

        if verbose: print('num_topics: {}, metric: {}, metric values: {}'.format(num_topics, metric, metric_values))

    if display_plot:

        plt.plot(range(topics_start, topics_limit, topics_step), metric_values)
        plt.xlabel('Num Topics')
        plt.ylabel('{} score'.format(metric))
        # plt.legend(('metric'), loc='best')
        plt.show()

    return model_list, metric_values

test = LDACalibration(topics_start=1, topics_limit=5, topics_step=1, dataframecolumn=df_articles['Article_sentence_nouns_cleaned'], topn=10, num_words=10, metric='coherence', verbose=True)

