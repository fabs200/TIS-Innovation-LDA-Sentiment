import pandas, time
from nltk.corpus import stopwords
from python.ConfigUser import path_processedarticles
from python.ProcessingFunctions import *

# Specify POStag type
POStag_type = 'NN'

start_time0 = time.process_time()

# Read in file with articles from R-Skript ProcessNexisArticles.R
df_articles = pandas.read_feather(path_processedarticles + 'feather/auto_articles_withbattery.feather')

######
# TEMP keep first 100 articles
df_articles = df_articles[df_articles['ID']<10]
######

# convert all words to lower case
df_articles['Article'] = [i.lower() for i in df_articles['Article']]

# Drop duplicates
df_articles.drop_duplicates(subset=['Article', 'Date'], inplace=True)
df_articles.drop_duplicates(subset=['Headline'], inplace=True)

# Remove text which defines end of articles
for splitstring in ['graphic', 'foto: classification language', 'classification language']:
    df_articles['Article'] = df_articles['Article'].str.split(splitstring).str[0]
df_articles['Article'] = [re.compile(r'(kommentar seite \d+)').sub(
    lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]
df_articles['Article'] = [re.compile(r'deliverynotification').sub(
    lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]

# Make Backup
df_articles['Article_backup_raw'] = df_articles['Article']

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
df_articles['Article'] = df_articles['Article'].str.replace("'", '').str.replace("\\", '').str.replace('"', '').str.replace('+', '')

### Put Articles into a nested list in list so we can apply same fcts as we do to sentences and paragraphs
df_articles['Article'] = df_articles['Article'].apply(ArticlesToLists)

### Remove additional words, remove links and emails
drop_words = ['taz', 'dpa', 'de', 'foto', 'webseite', 'herr', 'interview', 'siehe grafik', 'vdi nachrichten', 'vdi',
              'reuters', ' mid ', 'sz-online']
df_articles['Article'] = df_articles['Article'].apply(lambda x: WordRemover(x, dropWords=drop_words))
df_articles['Article'] = df_articles['Article'].apply(lambda x: LinkRemover(x))
df_articles['Article'] = df_articles['Article'].apply(lambda x: MailRemover(x))

end_time0 = time.process_time()

print('timer0: Elapsed time is {} seconds.'.format(round(end_time0-start_time0, 2)))

start_time1 = time.process_time()




### Remove punctuation except hyphen and apostrophe between words, special characters
df_articles['Article'] = df_articles['Article'].apply(lambda x: SpecialCharCleaner(x))


# Fork textbody for LDA GetTopics and flatten to string
df_articles['Article_backup'] = df_articles['Article']
# ToDo: convert list to string


# not solving hyphenation as no universal rule found

### POS tagging and tokenize words in sentences (time-consuming!) and run Lemmatization (Note: word get tokenized)
df_articles['Article_nouns'] = df_articles['Article'].apply(lambda x: POStagger(x, POStag=POStag_type))
df_articles['Article_nouns'] = df_articles['Article_nouns'].apply(lambda x: Lemmatization(x))

# Cleaning: drop stop words, drop if sentence contain only two words or less
df_articles['articles{}_for_lda'.format(POStag_type)] = df_articles['Article_nouns'].apply(TokensCleaner,
                                                                          minwordinsent=2,
                                                                          minwordlength=2)

### Export data to csv (will be read in again in LDACalibration.py)
df_articles[['ID_incr', 'ID', 'Date', 'articles{}_for_lda'.format(POStag_type)]].to_csv(
    path_processedarticles + 'csv/articles_for_lda_{}.csv'.format(POStag_type), sep='\t', index=False)

### Export as Excel and add Raw Articles
pandas.DataFrame(df_articles, columns=['Article_backup', 'articles{}_for_lda'.format(POStag_type)]).to_excel(
    path_processedarticles + 'articles_for_lda_{}.xlsx'.format(POStag_type))

df_articles['Temp'] = df_articles['articles{}_for_lda'.format(POStag_type)]


# Make Longfile
df_long_articles = df_articles.Temp.apply(pandas.Series)\
    .merge(df_articles[['ID_incr']], left_index = True, right_index = True)\
    .melt(id_vars = ['ID_incr'], value_name = 'articles_{}_for_lda'.format(POStag_type))\
    .dropna(subset=['articles_{}_for_lda'.format(POStag_type)])\
    .merge(df_articles[['ID_incr', 'Date', 'Newspaper']], how='inner', on='ID_incr')


df_articles['Temp'] = df_articles['Article_backup']

df_long_articles2 = df_articles.Temp.apply(pandas.Series)\
    .merge(df_articles[['ID_incr']], left_index = True, right_index = True)\
    .melt(id_vars = ['ID_incr'], value_name = 'article_text')\
    .dropna(subset=['article_text'])\
    .merge(df_articles[['ID_incr', 'Date', 'Newspaper']], how='inner', on='ID_incr')



### Export longfile to csv (will be read in later)
df_long_articles.to_csv(path_processedarticles + 'csv/articles_for_lda_{}_l.csv'.format(POStag_type), sep='\t', index=False)
df_long_articles.to_excel(path_processedarticles + 'articles_for_lda_{}_l.xlsx'.format(POStag_type))

df_long_articles2.to_csv(path_processedarticles + 'csv/articles_text.csv'.format(POStag_type), sep='\t', index=False)
df_long_articles2.to_excel(path_processedarticles + 'articles_text.xlsx'.format(POStag_type))



end_time2 = time.process_time()

print('timer2: Elapsed time is {} seconds.'.format(round(end_time2-start_time2, 2)))

print('Overall elapsed time is {} seconds.'.format(round(end_time2-start_time0, 2)))

# Clean up to keep RAM small
#del df_articles, df_long_articles, stopwords, drop_words

###
