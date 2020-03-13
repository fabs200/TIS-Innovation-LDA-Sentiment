import pandas, re
from nltk.corpus import stopwords
from python.ConfigUser import path_processedarticles
from python.ProcessingFunctions import Sentencizer, SentenceCleaner, SentencePOStagger, NormalizeWords, SentenceWordRemover, \
    SentenceLinkRemover, SentenceMailRemover, DateRemover, SentenceCleanTokens, NumberComplexRemover, SentenceLemmatizer

# Read in file with articles from R-Skript ProcessNexisArticles.R
df_articles = pandas.read_feather(path_processedarticles + 'feather/autofiles_withbattery.feather')

# convert all words to lower case
df_articles['Article'] = [i.lower() for i in df_articles['Article']]

# Drop duplicates
df_articles.drop_duplicates(subset=['Article', 'Date'], inplace=True)
df_articles.drop_duplicates(subset=['Headline'], inplace=True)

# Remove text which defines end of articles
df_articles['Article'] = df_articles['Article'].str.split('graphic').str[0]
df_articles['Article'] = df_articles['Article'].str.split('foto: classification language').str[0]
df_articles['Article'] = df_articles['Article'].str.split('classification language').str[0]
cutpage = re.compile(r'(kommentar seite \d+)')
cutpage2 = re.compile(r'deliverynotification')
df_articles['Article'] = [cutpage.sub(lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]
df_articles['Article'] = [cutpage2.sub(lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]

# Make Backup
df_articles['Article_backup'] = df_articles['Article']

# Create id increasing (needed to merge help files later)
df_articles.insert(0, 'ID_incr', range(1, 1 + len(df_articles)))

# Normalize Words (preserve words by replacing by synonyms and write full words instead abbrev.)
df_articles['Article'] = df_articles['Article'].apply(lambda x: NormalizeWords(x))

### Numbers in Text
# First, remove dates of the format: 20. Februar, e.g.
df_articles['Article'] = df_articles['Article'].apply(lambda x: DateRemover(x))
# Second, remove all complex combinations of numbers and special characters
df_articles['Article'] = df_articles['Article'].apply(lambda x: NumberComplexRemover(x)) # TODO: check again
# Third, remove all remaining numbers
df_articles['Article'] = df_articles['Article'].str.replace('\d+', '')

### Special Characters
df_articles['Article'] = df_articles['Article'].str.replace("'", '')

### Split sentence-wise
df_articles['Article_sentence'] = df_articles['Article'].apply(lambda x: Sentencizer(x))

### Remove additional words, remove links and emails
drop_words = ['taz', 'dpa', 'de', 'foto', 'webseite', 'herr', 'interview', 'siehe grafik', 'vdi nachrichten', 'vdi',
              'reuters', ' mid ', 'sz-online']
df_articles['Article_sentence'] = df_articles['Article_sentence'].apply(lambda x: SentenceWordRemover(x,
                                                                                                      dropWords=drop_words))
df_articles['Article_sentence'] = df_articles['Article_sentence'].apply(lambda x: SentenceLinkRemover(x))
df_articles['Article_sentence'] = df_articles['Article_sentence'].apply(lambda x: SentenceMailRemover(x))

### Remove punctuation except hyphen and apostrophe between words, special characters
df_articles['Article_sentence'] = df_articles['Article_sentence'].apply(lambda x: SentenceCleaner(x))

# not solving hyphenation as no univeral rule found

### POS tagging and tokenize words in sentences (time-consuming!) and run Lemmatization (Note: word get tokenized)
df_articles['Article_sentence_nouns'] = df_articles['Article_sentence'].apply(lambda x: SentencePOStagger(x,
                                                                                                          POStag='NN'))
df_articles['Article_sentence_nouns'] = df_articles['Article_sentence_nouns'].apply(lambda x: SentenceLemmatizer(x))

# Cleaning: drop stop words, drop if sentence contain only two words or less
df_articles['Article_sentence_nouns_cleaned'] = df_articles['Article_sentence_nouns'].apply(SentenceCleanTokens,
                                                                                            minwordinsent=2,
                                                                                            minwordlength=2)
pandas.DataFrame(df_articles, columns=['Article_backup', 'Article_sentence_nouns_cleaned']).to_excel(
    path_processedarticles + "Article_sentence_nouns_cleaned.xlsx")

# # Export data to csv (will be read in again in LDAArticles.py)
df_articles[['ID_incr', 'ID', 'Date', 'Article_sentence_nouns_cleaned']].to_csv(
    path_processedarticles + 'csv/sentences_for_lda_analysis.csv', sep='\t', index=False)

# Clean up to keep RAM small
del df_articles, cutpage, cutpage2, stopwords, drop_words

###
