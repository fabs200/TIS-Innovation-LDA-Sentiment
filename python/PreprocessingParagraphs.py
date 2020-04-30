import pandas, time
from nltk.corpus import stopwords
from python.ConfigUser import path_processedarticles
from python._ProcessingFunctions import *
from python.params import params as p

# unpack POStag type
POStag_type = p['POStag_type']

start_time0 = time.process_time()

# Read in file with articles from R-Skript ProcessNexisArticles.R
df_articles = pandas.read_feather(path_processedarticles + 'feather/auto_paragraphs_withbattery.feather')

######
# TEMP keep first x articles
# df_articles = df_articles[df_articles['Art_ID']<10]
######

# Write all paragraphs into a list of lists
paragraphs_list = []
for name, group in df_articles.groupby('Art_ID'):
    paragraphs_list.append(group['Paragraph'].to_list())

# create new df containing list of paragraphs in a cell with Art_ID to merge to above df
df_articles_lists = pandas.DataFrame({'Art_ID': list(range(1, len(paragraphs_list)+1)),
                                        'paragraph': paragraphs_list},
                                       columns=['Art_ID', 'paragraph'])

# Before merging, for each Art_ID keep one row (to keep dataset small when merging)
df_articles_TEMP = df_articles[df_articles['Par_ID']==1]

# Merge df_articles_lists to df_articles
df_articles = df_articles_TEMP.merge(df_articles_lists, left_on='Art_ID', right_on='Art_ID')

# Drop unnecessary vars
df_articles = df_articles.drop(columns=['Par_ID', 'Paragraph'])

# Make Backup
df_articles['paragraph_backup'] = df_articles['paragraph']

# convert all words to lower case
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: [i.lower() for i in x])

# Drop duplicates (TEMP do not drop any observations, when merging to other long-files, only join inner)
# df_articles.drop_duplicates(subset=['paragraph', 'Date'], inplace=True)
df_articles.drop_duplicates(subset=['Headline'], inplace=True)

# Remove text which defines end of articles
splittingstrings = ['graphic', 'foto: classification language', 'classification language', 'kommentar seite ',
                    'publication-type', 'classification', 'language: german; deutsch', 'bericht - seite '] #TODO: @Daniel 'classification' hier auch?
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: ParagraphSplitter(x, splitAt=splittingstrings))

# Create id increasing (needed to merge help files later)
df_articles.insert(0, 'ID_incr', range(1, 1 + len(df_articles)))

# Normalize Words (preserve words by replacing by synonyms and write full words instead abbrev.)
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: [NormalizeWords(i) for i in x])

### Numbers in Text
# First, remove dates of the format: 20. Februar, e.g.
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: [DateRemover(i) for i in x])
# Second, remove all complex combinations of numbers and special characters
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: [NumberComplexRemover(i) for i in x]) # TODO: check again
# Third, remove all remaining numbers
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: [i.replace('\d+', '') for i in x])

### Special Characters
drop_specchars = ["'", "\\", '"', '+']
for s in drop_specchars:
    df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: [i.replace("{}".format(s), '') for i in x])

### Remove additional words, remove links and emails
drop_words = ['taz', 'dpa', 'de', 'foto', 'webseite', 'herr', 'interview', 'siehe grafik', 'vdi nachrichten', 'vdi',
              'reuters', ' mid ', 'sz-online']
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: WordRemover(x, dropWords=drop_words))
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: LinkRemover(x))
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: MailRemover(x))

end_time0 = time.process_time()
print('timer0: Elapsed time is {} seconds.'.format(round(end_time0-start_time0, 2)))

start_time1 = time.process_time()

### Fork paragraphs for Sentiment Analysis
df_articles['paragraphs_text'] = df_articles['paragraph']

### Remove punctuation except hyphen and apostrophe between words, special characters
df_articles['paragraph'] = df_articles['paragraph'].apply(lambda x: SpecialCharCleaner(x))

# not solving hyphenation as no univeral rule found

### POS tagging and tokenize words in sentences (time-consuming!) and run Lemmatization (Note: word get tokenized)
df_articles['paragraph_nouns'] = df_articles['paragraph'].apply(lambda x: POStagger(x, POStag=p['POStag_type']))
df_articles['paragraph_nouns'] = df_articles['paragraph_nouns'].apply(lambda x: Lemmatization(x))

# Cleaning: drop stop words, drop if sentence contain only two words or less # TODO: calibrate later
df_articles['paragraphs_{}_for_lda'.format(POStag_type)] = df_articles['paragraph_nouns'].apply(TokensCleaner,
                                                                                           minwordinsent=p['minwordinsent'],
                                                                                           minwordlength=p['minwordlength'],
                                                                                                drop=False)

### Export data to csv (will be read in again in LDACalibration.py)
# df_articles[['ID_incr', 'Art_ID', 'Date', 'paragraphs_{}_for_lda'.format(POStag_type)]].to_csv(
#     path_processedarticles + 'csv/paragraphs_for_lda_{}.csv'.format(POStag_type), sep='\t', index=False)

### Export as Excel and add Raw Articles
# pandas.DataFrame(df_articles, columns=['paragraph_backup', 'paragraphs_{}_for_lda'.format(POStag_type)]).to_excel(
#     path_processedarticles + 'paragraphs_for_lda_{}.xlsx'.format(POStag_type))

# Make long file
df_long = df_articles.paragraphs_text.apply(pandas.Series)\
    .merge(df_articles[['ID_incr']], left_index = True, right_index = True)\
    .melt(id_vars = ['ID_incr'], value_name = 'paragraphs_text')\
    .dropna(subset=['paragraphs_text'])\
    .merge(df_articles[['ID_incr', 'Art_ID', 'Date', 'Newspaper']], how='inner', on='ID_incr')\
    .merge(df_articles['paragraphs_{}_for_lda'.format(POStag_type)].apply(pandas.Series)\
    .merge(df_articles[['ID_incr']], left_index = True, right_index = True)\
    .melt(id_vars = ['ID_incr'], value_name = 'paragraphs_{}_for_lda'.format(POStag_type))\
    .dropna(subset=['paragraphs_{}_for_lda'.format(POStag_type)]))\
    .drop(columns=['ID_incr', 'variable'])

# Generate Par_ID
df_long['Par_ID'] = df_long.groupby(['Art_ID']).cumcount()+1

# Sort columns
df_long = df_long[['Art_ID', 'Par_ID', 'Newspaper', 'Date', 'paragraphs_text', 'paragraphs_{}_for_lda'.format(POStag_type)]]

### Export longfile to csv (will be read in later)
df_long.to_csv(path_processedarticles + 'csv/paragraphs_for_lda_{}_l.csv'.format(POStag_type),
               sep='\t', index=False)
df_long.to_excel(path_processedarticles + 'paragraphs_for_lda_{}_l.xlsx'.format(POStag_type))

end_time1 = time.process_time()
print('timer1: Elapsed time is {} seconds.'.format(round(end_time1-start_time1, 2)))
print('Overall elapsed time is {} seconds.'.format(round(end_time1-start_time0, 2)))

###
