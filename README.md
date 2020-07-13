# TIS Innovation
project topic modelling and sentiment analysis of TIS innovation electic cars

## Disclaimer
Disclaimer: The author posed the following code for academic purposes
and an illustration of Selenium only. Scraping LexisNexis may be a
violation of LexisNexis user policy. Use at your own legal risk.

## Tasks
- [x] Daniel: add config_user.py to your project, set up your paths
- [x] Remove pw
- [x] check for incorrectly not tagged words in POS (e.g. 'EU')
- [x] check frequencies of topics
- [x] Implement word-doc tasks (13.02 Email)
- [x] implement spacy.lang.de Sentence Boundary Detection
- [x] prepare articles sentence-level
- [x] test stop-word needed or not, or token.tag_.startswith('NN') enough
- [x] Local LDA
- [x] implement sentiment analysis sepl
- [x] frequency distribution sentiment analysis negation words
- [x] make long file
- [x] check LDA_TEST_get_topics.py
  - [x]   coherence score, (grid search, alpha, beta, set random seed)
  - [x]   Leibler divergence (Li et al. 2019)
  - [x]   or 'auto'/'symmetric' as lda args (eyeballing)
- [x] update and adjust SePL by manneg phrs
- [x] add manneg phrs to ProcessSePLphrases(raw_sepl_phrase) and return it, too
- [x] final programming routine to create final analysis df long (colums: sentence/parag/article, topic distr, dom. topic, avg sent, med sent, sd sent, n sent(/phr), list phrs)
- [x] create one Analysis file (lda + sentiment analysis)
- [x] config_LDA.py: set up dictionary with parameters
- [x] Test lds on Paragraph-level
- [x] when LDA_TEST_get_topics.py checked -> set up code
- [x] topic modelling vizualization ([link](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/), word doc)
- [x] clean up data files in cloud
- [x] additional doc types
- [x] plot: lda time series
- [x] Get large dataset
- [x] Check Frequency:
    - [x] all SePL phrs
    - [x] SePL phrs by topics
- [x] TD-IDF lda model (instead of doc2bow)
- [x] Drop duplicate articles based on similarity 
- [x] write lda as class()
- [x] Sentiment Analysis accuracy
