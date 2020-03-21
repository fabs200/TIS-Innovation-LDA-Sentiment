# Help functions for LDA and Sentiment Analysis
import spacy
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from python.ConfigUser import path_processedarticles
import pandas

def Load_SePL():
    """
    Reads in SePL file, prepares phrases and sorts them; this is required be be run before MakeCandidates() and 
    GetSentiments()
    """
    # Read in SePL
    df_sepl = pandas.read_csv(path_processedarticles + 'SePL/SePL_v1.1.csv', sep=';')

    # convert all words to lower case
    df_sepl['phrase'] = [i.lower() for i in df_sepl['phrase']]

    df_sepl['phrase_sorted'] = df_sepl['phrase'].apply(lambda x: ' '.join(sorted(x.split())))
    print('SePL file loaded')

    return df_sepl

nlp2 = spacy.load('de_core_news_md', disable=['ner', 'parser'])

def MakeCandidates(sent, df_sepl=None, get='candidates', verbose=False, negation_list=None):
    """
    prepares a nested list of candidates, make sure df_sepl is loaded (run Load_SePL() before)
    :param sent: input is a full sentence as string
    :param df_sepl: load df_sepl via Load_SePL()
    :param verbose: display
    :param get: default 'candidates', else specify 'negation' to geht same list in lists but only negation words
    :param negation_list: specifiy list with negation words to identify negated sentences; negation_list must not specified
    :return: nested list of lists where each nested list is separated by the POS tag $,
    """

    # split sentence by comma, write in list and read in via nlp2()
    sent = sent.split(',')
    sent = [nlp2(s) for s in sent]
    candidates = []

    if get == 'candidates':

        # Loop over sentence parts and check whether a word is a noun/verb/adverb/adjective and append it as candidate
        for s in sent:
            c = []
            # loop over tokens in sentences, get tags and prepare
            for token in s:
                if verbose: print('token:', token.text, '->', token.tag_)
                if token.tag_.startswith(('NN', 'V', 'ADV', 'ADJ')):
                    if df_sepl['phrase'].str.contains(r'(?:\s|^){}(?:\s|$)'.format(token)).any():
                        c.append(token.text)
            candidates.append(c)

        if verbose: print('final candidates:', candidates)

    if get == 'negation':

        if negation_list is None:
            # Rill (2016)
            # negation_list = ['nicht', 'kein', 'nichts', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'niemanden',
            #                  'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']
            # Rill + Wiegant et al. (2018)
            negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                             'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']
            # TODO: check further negation words in literature

        # loop over sentence parts and check whether a word is contained in negotion_list, if yes, append to candidates
        for s in sent:
            c = []
            # loop over tokens in sentence part
            for token in s:
                if verbose: print(token.text, token.tag_)
                if (token.text in negation_list):
                # if (token.tag_.startswith(('PIAT', 'PIS', 'PTKNEG'))) or (token.text in negation_list):
                    c.append(token.text)
            candidates.append(c)
        if verbose: print('final negations:', candidates)

    return candidates

def GetSentiments(candidates, df_sepl=None, verbose=False):
    """
    reads in candidates (list in list), retrieves sentiment scores (sentiment_scores), returns them and the opinion
    relevant terms (tagged_phr), make sure df_sepl is loaded (run Load_SePL() before)
    :param candidates: list in list with POS tagged words
    :param df_sepl: load df_sepl via Load_SePL()
    :param verbose: display
    :return: [sentiment_scores], [tagged_phr]
    """

    final_sentiments, final_phrs, tagged_phr_list = [], [], []
    # loop over candidates and extract sentiment score according to Rill (2016): S.66-73, 110-124
    for c in candidates:
        c_sentiments, c_phrs = [], []
        # loop over each word in nested candidate list
        for word in c:
            stack = []
            index = c.index(word)
            if verbose: print('\n###### word:', word, 'index:', index, 'candidates:', c, '######')

            # check whether candidate is contained in SePL, if yes, get left and right neighbors
            if df_sepl['phrase_sorted'].str.contains(word).any():
                stack.append(word)
                if verbose: print(word, '|| stack without neighbours:', stack)
                for i in c[index+1:]:
                    stack.append(i)
                    if verbose: print(word, '|| stack with right neighbours:', stack)

                # select slice of left neigbours and reverse it with second bracket
                for x in c[:index][::-1]:
                    stack.append(x)
                    if verbose: print (word, '|| stack with left neighbours:', stack)

                if verbose: print('final stack:', stack)

                # loop over stack and check whether word in SePL, if not, delete el from stack, if yes, extract sentiment,
                # delete el from stack and continue with remaining el in stack
                while len(stack)>0:
                    phr = sorted(stack)
                    phr_string = ' '.join(phr)
                    if verbose: print('phr_string:', phr_string)

                    # if el of stack found in SePL, extract sentiment and save phrases
                    if (df_sepl['phrase_sorted'] == phr_string).any() and phr_string not in c_phrs and phr_string not in tagged_phr_list:
                        # extract sentiment
                        sentiment_score = df_sepl.loc[df_sepl['phrase_sorted'] == phr_string, 'opinion_value'].item()
                        c_sentiments.append(sentiment_score)
                        if verbose: print('phrase found! sentiment is', sentiment_score)
                        # save phr
                        c_phrs.append(phr_string)
                        tagged_phr_list = phr_string.split()
                        break

                    # if el of stack not found, delete it and continue with next el in stack
                    else:
                        if verbose: print('deleting', stack[-1])
                        del stack[-1]

        # gather all extracted sentiments and phrases
        final_sentiments.append(c_sentiments)
        final_phrs.append(c_phrs)

    if verbose: print('final list with sentiments:', final_sentiments)
    if verbose: print('final list of phrs:', final_phrs)

    return final_sentiments, final_phrs
