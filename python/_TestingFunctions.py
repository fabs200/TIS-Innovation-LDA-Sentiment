"""
Testing functions
"""
from python.ConfigUser import path_project
import pandas, xlsxwriter
from python._AnalysisFunctions import MakeCandidates, ReadSePLSentiments, Load_SePL

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

def ExtractNegSeplPhr(sepl_phrase, negation_candidates, negation_list=None):
    """
    Check Chase II words from ProcessSentimentScores()
    :param sepl_phrase: GetSentiments(...)[1], here are all words which are in SePL
    :param negation_candidates: MakeCandidates(..., get='negation')
    :return: 1 sentiment score
    """

    if negation_list is None:
        negation_list = ['nicht', 'kein', 'nichts', 'kaum', 'ohne', 'niemand', 'nie', 'nie mehr', 'niemals', 'gegen',
                         'niemanden', 'keinesfalls', 'keineswegs', 'nirgends', 'nirgendwo', 'mitnichten']

    # Loop over each sentence part and access each list (sepl_word/negation_candidates/sentimentscores) via index
    sepl_phrase_inv_neg = []
    for i in range(0, len(sepl_phrase)):

        # Check whether sepl_word in sentence part is contained in negation_list, if yes, set flag to True
        # if sepl_phrase[i]:
        if sepl_phrase[i] and negation_candidates[i]:

            # write as str
            sepl_string = sepl_phrase[i][0]
            sepl_neg_string = negation_candidates[i][0]

            # set up flags
            seplphr, seplphrneg = False, False

            # check whether negation word in sepl_string, in sepl_neg_string
            for word in sepl_string.split():
                if word in negation_list: seplphr = True
            for word in sepl_neg_string.split():
                if word in negation_list: seplphrneg = True

            # Condition Case II
            if not seplphr and seplphrneg:
                sepl_phrase_inv_neg.append(sepl_neg_string + ' ' + sepl_string)
        else:
            continue

    # Flatten list
    # flatsentimentscores = [element for sublist in sentimentscores for element in sublist]

    return sepl_phrase_inv_neg



def GetNegatedSepl(listOfSents, df_sepl=df_sepl, phronly=False):
    """
    SePL does not include a negated phrase for every word in the list (p.72 f.). To circumvent this problem:
    This function is a wrapper which first loads the candidates and negation_candidates as nested lists and passes them
    to ReadSePLSentiments(). This fct extracts sentimentscore and sepl_phrase. Least, we pass further to
    ExtractNegSeplPhr() which then checks whether or not it is manually negated by our rule (if negation word in
    negation_list contained, then invert sentimentscore). We want to investigate which and how many negated phrases are
    missing in the SePL list and if needed extend frequent negated phrases.
    apply this fct to the column 'Article_sentiment_sentences' e.g. and run via pandas .apply(lambda x:...) to each row.
    :param listOfSents: nested list in list
    :param df_sepl: Pandas dataframe, load it via df_sepl=LoadSePL()
    :param phronly: True, False, set to False if you want to get only also empty lists such that one can trace the location.
    :return: nested list in list with manually negated sepl_phrases as elements
    """

    negSeplList = []

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
        sentimentscores, sepl_phrase = ReadSePLSentiments(candidates, df_sepl=df_sepl)

        """
        Extract manually negated phrases not in negated form SePL (but in positive)
        """
        try:
            temp_mannegphr = ExtractNegSeplPhr(sepl_phrase, negation_candidates)
        except:
            temp_mannegphr = []

        if phronly and temp_mannegphr:
            negSeplList.append(temp_mannegphr)
        if not phronly:
            negSeplList.append(temp_mannegphr)

    return negSeplList

