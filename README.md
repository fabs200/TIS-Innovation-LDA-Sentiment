# Media Analysis
project topic modelling and sentiment analysis

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
- [ ] check LDA_TEST_get_topics.py
- [ ] set up lda evaluation (e.g. pyLDAvis) of pre- and postprocessing
      of text
  - [ ]   coherence score, (grid search, alpha, beta)
  - [ ]   Leibler divergence (Li et al. 2019)
  - [ ]   or 'auto'/'symmetric' as lda args
- [ ] plot: lda time series
- [ ] Get large dataset
- [ ] Local LDA
- [ ] implement sentiment analysis sepl


## Tasks later
- [ ] config_LDA.py: set up dictionary with parameters
- [ ] when LDA_TEST_get_topics.py checked -> set up code
- [ ] create guidedLDA.py, config_guidedLDA.py
- [ ] Lda multicore for faster processing
- [ ] Bigram/trigram lda model
- [ ] TD-IDF lda model (instead of doc2bow)
- [ ] Drop duplicate articles based on similarity 
- [ ] Test lds on Pragraph-level
