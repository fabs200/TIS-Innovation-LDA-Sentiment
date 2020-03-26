
########################
########################

# TODo: check function for coherence score, should be compatible to EstimateLDA function
# TODO: test alpha = 'auto' and eta ='auto' in lda model and add to compute_coherence function
# ToDO: Function to compute Kullback-Leibler divergence between topics

#u_mass coherence measure
from gensim.models.coherencemodel import CoherenceModel
lda_nouns_cm = CoherenceModel(model=lda_nouns, corpus=corpus_nouns, dictionary=dict_nouns, coherence="u_mass")
print(lda_nouns_cm.get_coherence())

##we use coherence measure c_v as suggested by Röder et al. 2015, because it has the highest correlation with human interpretability



##with coherence measure: u_mass



def compute_coherence_values(dictionary, corpus, texts, id2word, topics_limit, topics_start, topics_step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    TODO: add random state to get same results

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(topics_start, topics_limit, topics_step):
        model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, alpha='auto', eta='auto', random_state=203495)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print('num_topics:', num_topics, 'coherence:', coherencemodel.get_coherence())
    return model_list, coherence_values

start, limit, step = 1, 10, 1

model_list, coherence_values = compute_coherence_values(dictionary=dict_nouns, id2word=id2word_nouns, corpus=corpus_nouns, texts=nouns, topics_start=start, topics_limit=limit, topics_step=step)
# Show graph
import matplotlib.pyplot as plt
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
