import pandas, time
from nltk.corpus import stopwords
from python.ConfigUser import path_data
from python._ProcessingFunctions import *
from python.params import params as p

# unpack POStag type
POStag_type = p['POStag']

start_time0 = time.process_time()

# Read in file with articles from R-Skript ProcessNexisArticles.R
df_articles = pandas.read_feather(path_data + 'feather/auto_articles_withbattery.feather')

######
# TEMP keep first x articles
# df_articles = df_articles[df_articles['Art_ID']<100]
######

# convert all words to lower case
df_articles['Article'] = [i.lower() for i in df_articles['Article']]

# Drop duplicates
df_articles.drop_duplicates(subset=['Article', 'Date'], inplace=True)
df_articles.drop_duplicates(subset=['Headline'], inplace=True)

# Remove text which defines end of articles
for splitstring in ['graphic', 'foto: classification language', 'classification language', 'kommentar seite ',
                    'publication-type', 'classification', 'language: german; deutsch', 'bericht - seite ']: #TODO: @Daniel 'classification' hier auch?
    df_articles['Article'] = df_articles['Article'].str.split(splitstring).str[0]
df_articles['Article'] = [re.compile(r'(kommentar seite \d+)').sub(
    lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]
df_articles['Article'] = [re.compile(r'deliverynotification').sub(
    lambda m: (m.group(1) if m.group(1) else " "), x) for x in df_articles['Article'].tolist()]

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
df_articles['Article'] = df_articles['Article'].str.replace("'", '').str.replace("\\", '').str.replace('"', '').str.replace('+', '')

### Split sentence-wise
df_articles['sentence'] = df_articles['Article'].apply(lambda x: Sentencizer(x))

### Remove additional words, remove links and emails
drop_words = ['taz', 'dpa', 'de', 'foto', 'webseite', 'herr', 'interview', 'siehe grafik', 'vdi nachrichten', 'vdi',
              'reuters', ' mid ', 'sz-online']
df_articles['sentence'] = df_articles['sentence'].apply(lambda x: WordRemover(x, dropWords=drop_words))
df_articles['sentence'] = df_articles['sentence'].apply(lambda x: LinkRemover(x))
df_articles['sentence'] = df_articles['sentence'].apply(lambda x: MailRemover(x))

print('timer0: Elapsed time is {} seconds.'.format(round(time.process_time()-start_time0, 2)))
start_time1 = time.process_time()

### Fork sentences for Sentiment Analysis
df_articles['sentences_for_sentiment'] = df_articles['sentence'].apply(lambda x: CapitalizeNouns(x))

### Remove punctuation except hyphen and apostrophe between words, special characters
df_articles['sentence'] = df_articles['sentence'].apply(lambda x: SpecialCharCleaner(x))

print('timer1: Elapsed time is {} seconds.'.format(round(time.process_time()-start_time1, 2)))
start_time2 = time.process_time()

# not solving hyphenation as no universal rule found

### POS tagging and tokenize words in sentences (time-consuming!) and run Lemmatization (Note: word get tokenized)
df_articles['sentence_nouns'] = df_articles['sentence'].apply(lambda x: POSlemmatizer(x, POStag=p['POStag']))
df_articles['sentence_nouns'] = df_articles['sentence_nouns'].apply(lambda x: Lemmatization(x))

# Cleaning: drop stop words, drop if sentence contain only two words or less # TODO: calibrate later
df_articles['sentences_{}_for_lda'.format(POStag_type)] = df_articles['sentence_nouns'].apply(TokensCleaner,
                                                                                           minwordinsent=p['minwordinsent'],
                                                                                           minwordlength=p['minwordlength'],
                                                                                              drop=False)

### Export data to csv
# df_articles[['ID_incr', 'Art_ID', 'Date', 'sentences_{}_for_lda'.format(POStag), 'sentences_for_sentiment']].to_csv(
#     path_data + 'csv/sentences_for_lda_{}.csv'.format(POStag), sep='\t', index=False)

### Export as Excel and add Raw Articles
# pandas.DataFrame(df_articles, columns=['Article_backup', 'sentences_{}_for_lda'.format(POStag)]).to_excel(
#     path_data + 'sentences_for_lda_{}.xlsx'.format(POStag))

# Make long file
df_long = df_articles.sentences_for_sentiment.apply(pandas.Series)\
    .merge(df_articles[['ID_incr']], left_index = True, right_index = True)\
    .melt(id_vars = ['ID_incr'], value_name = 'sentences_for_sentiment')\
    .dropna(subset=['sentences_for_sentiment'])\
    .merge(df_articles[['ID_incr', 'Art_ID', 'Date', 'Newspaper']], how='inner', on='ID_incr')\
    .merge(df_articles['sentences_{}_for_lda'.format(POStag_type)].apply(pandas.Series)\
    .merge(df_articles[['ID_incr']], left_index = True, right_index = True)\
    .melt(id_vars = ['ID_incr'], value_name = 'sentences_{}_for_lda'.format(POStag_type))\
    .dropna(subset=['sentences_{}_for_lda'.format(POStag_type)]))\
    .rename(columns={"variable": "Sent_ID"})\
    .drop(columns=['ID_incr'])

# replace everything in brackets from Newspaper
df_long['Newspaper'] = df_long.Newspaper.replace(to_replace='\([^)]*\)', value='', regex=True).str.strip()

# Sort columns
df_long = df_long[['Art_ID', 'Sent_ID', 'Newspaper', 'Date', 'sentences_for_sentiment', 'sentences_{}_for_lda'.format(POStag_type)]]

### Export longfile to csv (will be read in later)
df_long.to_csv(path_data + 'csv/sentences_for_lda_{}_l.csv'.format(POStag_type), sep='\t', index=False)
df_long.to_excel(path_data + 'sentences_for_lda_{}_l.xlsx'.format(POStag_type))

print('timer2: Elapsed time is {} seconds.'.format(round(time.process_time()-start_time2, 2)))
print('Overall elapsed time is {} seconds.'.format(round(time.process_time()-start_time0, 2)))

###
