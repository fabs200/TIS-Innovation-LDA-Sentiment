"""
Testing functions
"""
from python.ConfigUser import path_project
import pandas, xlsxwriter
from python.AnalysisFunctions import MakeCandidates, ReadSePLSentiments, Load_SePL

df_sepl = Load_SePL()

def ListToFreqDict(wordlist):
    """
    reads in a list, counts words, puts them into a dictionary ans sorts them reversed
    """
    wordfreq = [wordlist.count(p) for p in wordlist]
    help = dict(list(zip(wordlist, wordfreq)))
    # sort ascending
    # help = sorted(help.items(), key=lambda kv: kv[1])
    # sort descending
    help = sorted(help.items(), reverse=True, key=lambda kv: kv[1])
    return help


def ExportFreqDict(wordlist, path=path_project + 'data/', filename='frequency_wordlist.xlsx'):
    """
    exports the produced frequency-wordlist from ListToFreqDict to an excel file
    input: so list with tuples with the form [(,),(,),...]
    """
    workbook = xlsxwriter.Workbook(path + filename)
    worksheet = workbook.add_worksheet()
    row, col = 1, 0
    worksheet.write(0, 0, 'word')
    worksheet.write(0, 1, 'frequency')
    for el in wordlist:
        worksheet.write(row, col, el[0])
        worksheet.write(row, col + 1, el[1])
        row += 1
    workbook.close()


def ExtractNegatedSeplPhrases(sepl_phrase, negation_candidates, negation_list=None):
    """
    Check Chase II words from ProcessSentimentScores()
    :param sepl_phrase: GetSentiments(...)[1], here are all words which are in SePL
    :param negation_candidates: MakeCandidates(..., get='negation')
    :return: 1 sentiment score
    """

    # TODO: not finished

    if negation_list is None:
        negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                         'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']

    # Loop over each sentence part and access each list (sepl_word/negation_candidates/sentimentscores) via index
    for i in range(0, len(sepl_phrase)):

        exportlist = []
        # Check whether sepl_word in sentence part is contained in negation_list, if yes, set flag to True
        if sepl_phrase[i]:
            sepl_string = sepl_phrase[i][0]
            sepl_phrase_in_negation_list = False
            for word in sepl_string.split():
                if word in negation_list: sepl_phrase_in_negation_list = True
            # Condition Case II
            if not sepl_phrase_in_negation_list and set(negation_candidates[i]).intersection(negation_list).__len__():
                # Invert sentiment
                # sentimentscores[i][0] = -sentimentscores[i][0]
                exportlist.append(sepl_string)
        else:
            continue

    # Flatten list
    return [element for sublist in exportlist for element in sublist]


def GetNegatedSepl(listOfSents, df_sepl=df_sepl):

    for sent in listOfSents:

        """
        first step: identification of suitable candidates for opinionated phrases suitable candidates: nouns, adjectives, 
        adverbs and verbs
        """
        candidates = MakeCandidates(sent, df_sepl=df_sepl, get='candidates')
        negation_candidates = MakeCandidates(sent, df_sepl=df_sepl, get='negation')

        """
        second step: extraction of possible opinion-bearing phrases from a candidate starting from a candidate, 
        check all left and right neighbours to extract possible phrases. The search is terminated on a comma (POS tag $,), 
        a punctuation terminating a sentence (POS tag $.), a conjunction (POS-Tag KON) or an opinion-bearing word that is 
        already tagged. (Max distance determined by sentence lenght)
        If one of the adjacent words is included in the SePL, together with the previously extracted phrase, it is added to 
        the phrase.
        """
        raw_sentimentscores, raw_sepl_phrase = ReadSePLSentiments(candidates, df_sepl=df_sepl)

        """
        Extract self negated phrases not in SePL
        """
        return ExtractNegatedSeplPhrases(raw_sepl_phrase, negation_candidates)

