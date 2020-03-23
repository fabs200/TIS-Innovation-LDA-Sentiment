# Help functions for LDA and Sentiment Analysis
import spacy
from spacy.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from python.ConfigUser import path_processedarticles
import pandas
import math

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
    #ToDo: split sentences if POS_tag = conjunction
    sent = sent.split(',')
    sent = [nlp2(s) for s in sent]
    candidates = []

    if negation_list is None:
        # Rill (2016)
        # negation_list = ['nicht', 'kein', 'nichts', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'niemanden',
        #                  'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']
        # Rill + Wiegant et al. (2018)
        negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                         'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']
        # TODO: check further negation words in literature

    if get == 'candidates':

        # Loop over sentence parts and check whether a word is a noun/verb/adverb/adjective and append it as candidate
        for s in sent:
            c = []
            # loop over tokens in sentences, get tags and prepare
            for token in s:
                if verbose: print('token:', token.text, '->', token.tag_)
                if token.tag_.startswith(('NN', 'V', 'ADV', 'ADJ')) or token.text in negation_list:
                    if df_sepl['phrase'].str.contains(r'(?:\s|^){}(?:\s|$)'.format(token)).any():
                        c.append(token.text)
            candidates.append(c)

        if verbose: print('final candidates:', candidates)

    if get == 'negation':

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

def ReadSePLSentiments(candidates, df_sepl=None, verbose=False):
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
                    if (df_sepl['phrase_sorted'] == phr_string).any() and phr_string not in c_phrs and set(tagged_phr_list).intersection(phr).__len__() == 0:
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

def ExtractSentimentFrequency(candidates, negation, sentimentscores):
    """

    # TODO: Add aspect-specific word list based on most frequent opinion phrases via spacy POS tagger
    # TODO: Check most frequently negated opinion phrases (based on manually tagged Negation, e.g. 'kein', 'nicht', ...) if sentiment score is appropriate (*-1)

    :param candidates:
    :param negation:
    :param sentimentscores:
    :return:
    """

def ProcessSentimentScores(sepl_phrase, negation_candidates, sentimentscores, negation_list=None):
    """
    Process sentimentscores of sentence parts and return only one sentiment score per sentence

        # Case I:
        # 1. any word contained in negation_list
        # (2. sentimentscore is not empty)
        # -> do nothing

        # Case II:
        # 1. any word is NOT contained in negation_list
        # (2. sentimentscore is not empty)
        # 3. negation_candidates is not empty
        # -> Invert sentimentscore

    :param sepl_phrase: GetSentiments(...)[1], here are all words which are in SePL
    :param negation_candidates: MakeCandidates(..., get='negation')
    :param sentimentscores: GetSentiments(...)[0]
    :return: 1 sentiment score
    """

    if negation_list is None:
        negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                         'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']

    # Loop over each sentence part and access each list (sepl_word/negation_candidates/sentimentscores) via index
    for i in range(0, len(sepl_phrase)):

        # Check whether sepl_word in sentence part is contained in negation_list, if yes, set flag to True
        if sepl_phrase[i]:
            sepl_string = sepl_phrase[i][0]
            sepl_phrase_in_negation_list = False
            for word in sepl_string.split():
                if word in negation_list: sepl_phrase_in_negation_list = True
            # Condition Case II
            if not sepl_phrase_in_negation_list and set(negation_candidates[i]).intersection(negation_list).__len__():
                # Invert sentiment
                sentimentscores[i][0] = -sentimentscores[i][0]
        else:
            continue

    # Flatten list
    flatsentimentscores = [element for sublist in sentimentscores for element in sublist]

    # Average sentiment score
    if flatsentimentscores:
        averagescore = sum(flatsentimentscores) / len(flatsentimentscores)
    else:
        averagescore = []

    return averagescore

def ProcessSePLphrases(sepl_phrase):
    """
    Process sepl_phrases of sentence parts and return only one list with the opinion relevant words per sentence,
    drop empty nested lists

    :param sepl_phrase: GetSentiments(...)[1], here are all words which are in SePL
    :return: 1 sepl_word list
    """

    # Loop over sentence parts and append only non-empty lists
    processed_sepl_phrases = ([])
    for phrase in sepl_phrase:
        if phrase:
            for p in phrase:
                processed_sepl_phrases.append(p)
    return processed_sepl_phrases


def GetSentimentScores(listOfSents, df_sepl):
    """
    Run this function on each article (sentence- or paragraph-level) and get final sentiment scores.
    Include following function:

    1. Load_SePL() to load SePL
    2. MakeCandidates() to make candidates- and candidates_negation-lists
    3. ReadSePLSentiments() which reads in candidates- and candidates_negation-lists and retrieves sentiment scores
        from SePL
    4. ProcessSentimentScores() to process the retrieved sentiment scores and to return a unified score per sentence

    :param listOfSents: articles must be processed with ProcessSentsforSentiment()
    :return: return 1 value per Article, return 1 list with sentiments of each sentence, 1 list w/ opinion relev. words
    """

    listOfSentimentsscores, listOfsepl_phrases = [], []

    for sent in listOfSents:

        """
        first step: identification of suitable candidates for opinionated phrases suitable candidates: nouns, adjectives, 
        adverbs and verbs
        """
        candidates = MakeCandidates(sent, df_sepl, get='candidates')
        negation_candidates = MakeCandidates(sent, df_sepl, get='negation')

        """
        second step: extraction of possible opinion-bearing phrases from a candidate starting from a candidate, 
        check all left and right neighbours to extract possible phrases. The search is terminated on a comma (POS tag $,), 
        a punctuation terminating a sentence (POS tag $.), a conjunction (POS-Tag KON) or an opinion-bearing word that is 
        already tagged. (Max distance determined by sentence lenght)
        If one of the adjacent words is included in the SePL, together with the previously extracted phrase, it is added to 
        the phrase.
        """

        raw_sentimentscores, raw_sepl_phrase = ReadSePLSentiments(candidates, df_sepl)

        """
        third step: compare extracted phrases with SePL After all phrases have been extracted, they are compared with the 
        entries in the SePL. (everything lemmatized!) If no  match is found, the extracted Phrase is shortened by the last 
        added element and compared again with the SePL. This is repeated until a match is found.
        """

        # Make sure sepl_phrase, negation_candidates, sentimentscores are of same size
        assert len(raw_sepl_phrase) == len(raw_sentimentscores) == len(candidates) == len(negation_candidates)

        # export processed, flattened lists
        sentimentscores = ProcessSentimentScores(raw_sepl_phrase, negation_candidates, raw_sentimentscores)
        sepl_phrase = ProcessSePLphrases(raw_sepl_phrase)

        listOfSentimentsscores.append(sentimentscores)
        listOfsepl_phrases.append(sepl_phrase)

    # Calculate average sentiment score per article
    article_score = sum([x for x in listOfSentimentsscores if x != []]) / len([x for x in listOfSentimentsscores if x != []])

    return article_score, listOfSentimentsscores, listOfsepl_phrases

