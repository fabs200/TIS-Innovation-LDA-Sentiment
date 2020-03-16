import pandas
import spacy
from python.ConfigUser import path_processedarticles
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
#Dissertation Rill: S.66-73, 110-124


# Read in SePL
df_sepl = pandas.read_csv(path_processedarticles + 'SePL/SePL_v1.1.csv', sep=';')

# convert all words to lower case
df_sepl['phrase'] = [i.lower() for i in df_sepl['phrase']]


df_sepl['phrase_sorted'] = df_sepl['phrase'].apply(lambda x: ' '.join(sorted(x.split())))

#df_sepl['phrase_sorted'] = df_sepl['phrase_sorted'].apply(word_tokenize)
#todo tokenize SEPL


#todo lemmatize text from articels


sentence1 = 'anfangs war er mir sehr unsympathisch' #included in SePL: anfangs sehr unsympathisch
sentence2 = 'ich finde ihn ganz gut, aber ich bin nicht sehr begeistert von seinem auto' #negation not included in SePL

nlp2 = spacy.load('de_core_news_md', disable=['ner', 'parser'])

sentence1 = nlp2(sentence1)
sentence2 = nlp2(sentence2)


#first step: identification of suitable candidates for opinionated phrases
#suitable candidates: nouns, adjectives, adverbs and verbs

candidates = []
for token in sentence1:
    print(token.tag_)
    if token.tag_.startswith(('NN','V','ADV', 'ADJ')):
        if df_sepl['phrase'].str.contains(r'(?:\s|^){}(?:\s|$)'.format(token)).any():
            candidates.append(token.text)
print(candidates)

#second step: extraction of possible opinion-bearing phrases from a candidate
#starting from a candidate, check all left and right neighbours to extract possible phrases.
# The search is terminated on a comma (POS tag $,), a punctuation terminating a sentence (POS tag $.),
# a conjunction (POS-Tag KON) or an opinion-bearing word that is already tagged. (Max distance determined by sentence lenght)
#If one of the adjacent words is included in the SePL, together with the previously extracted phrase, it is added to the phrase.
stack = []
for word in candidates:
    index = candidates.index(word)
    #check if word is included in SePL
    print('for filter',word)
    #if df_sepl['phrase_sorted'].str.contains(r'(?:\s|^){}(?:\s|$)'.format(word)).any():
    if df_sepl['phrase_sorted'].str.contains(word).any(): #todo: other function for tokens
        stack.append(word)
        print(stack)
        for i in candidates[index+1:]:
            stack.append(i)
        for i in candidates[:index-1]:
            stack.append(i)
            print ('stack', stack)
        while len(stack)>0:
            phr = sorted(stack)
            phr_string = ' '.join(phr)
            print ('phr', phr)
            if df_sepl['phrase_sorted'].str.contains(phr_string).any():
                sentiment_phrase = df_sepl.loc[df_sepl['phrase_sorted'] == phr_string, 'opinion_value'].item()
                break
            else:
                print('deleting', stack[-1])
                del stack[-1]



#third step: compare extracted phrases with SePL
#After all phrases have been extracted, they are compared with the entries in the SePL. (everything lemmatized!)
#If no  match is found, the extracted Phrase is shortened by the last added element and compared again with the SePL.
#This is repeated until a match is found.

#####(delete later): Die extrahierte Phrase „ganz sehr begeistert“ ist so nicht in der SePL enthalten. Jedoch ist die um ein Wort kürzere Phrase „sehr begeistert“ enthalten.


# SePL does not include a negated phrase for every word in the list (p.72 f.)

