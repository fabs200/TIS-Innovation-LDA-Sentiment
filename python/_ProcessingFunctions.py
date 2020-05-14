import spacy, re, pandas, warnings
from textdistance import jaro
from spacy.lang.de import German
from spacy.tokenizer import Tokenizer
from germalemma import GermaLemma
from python.params import params as p

"""
------------------------
_ProcessingFunctions.py
------------------------
Create help functions to call when running scripts
"""

# TODO: Clean up POStagger to POSlemmatizer
# TODO: delete unnecessary fct (Lemmeatizer(), Lemmatization() because included in POSlemmatizer)

def MakeListInLists(string):
    """
    make lists nested in list, this is used for reading in exported, preprocessed articles and to prepare them
    in the appropriate format we need for running lda
    """
    listinlist = []
    for n in string:
        # help = []
        m = n.replace('[', '').replace(']', '').replace('\'', '').replace("'", "")
        m = m.split(', ')
        # TODO: Check later whether loop below is needed when running lda
        # for o in m:
        #     help.append(o)
        # listinlist.append(help[0:(len(help[0]) - 1)])
        listinlist.append(m)
    return listinlist


# function to flatten lists
FlattenList = lambda l: [element for sublist in l for element in sublist]

def ListtoString(list):
    return ''.join(list)

def GetUniqueStrings(list, threshold=.9, verbose=False):
    """
    input: list with strings
    param: threshold, verbose
    return: 2 lists, one with unique strings, one with unique indizes
    """
    list.sort(key=len)
    unique_flag = True
    unique_strings, unique_index = [], []
    for i, l1 in enumerate(list):
        unique_flag = True
        if verbose: print(i, l1)
        for j in range(i + 1, len(list)):
            l2 = list[j]
            similarity_index = jaro(l1, l2)
            if similarity_index >= threshold:
                # if similiar, don't append to unique lists
                if verbose: print('similar strings:\n[{}] {}\n[{}] {}\n'.format(i, l1, j, l2))
                unique_flag = False
                continue
            else:
                unique_flag = True
        if unique_flag:
            unique_index.append(i)
            unique_strings.append(l1)
    return unique_index, unique_strings


def RemoveBlankElements(list):
    return [x.strip() for x in list if x.strip()]


def DateRemover(string):
    """
    remove dates of the format: 25. februar, OR 10, mai OR 1juni OR 8./9. juni OR 8., 9. juni OR 8.-9. juni OR 8. 9. juni
    run first, before NumberComplexRemover()
    """
    for month in ['januar', 'februar', 'märz', 'april', 'mai', 'juni', 'juli', 'august', 'september', 'oktober',
                  'november', 'dezember']:
        string = re.sub('\d+([.]\s+|\s+|)({})'.format(month), ' {} '.format(month), string)
    string = re.sub('\d+([.]|[.]\s+|\s+|)jahrhundert', 'jahrhundert', string)
    return string


def NumberComplexRemover(string):
    """
    removes numbers in complex format, but not if a . is followed as it introduces the end of a sentence.
    run after DateRemover()
    Examples: 15.10 Uhr OR 3,5 bis 4 stunden. OR 100 000 euro. OR 20?000 förderanträge OR um 2025/2030 OR
    OR abc 18.000. a OR abc. 18.000. a OR abc 18. a  OR abc 7.8.14. a  OR abc 7. 14. 18. a OR abc 1970er. a
    OR abc 20?()/&!%000. a  OR abc 2,9-3,5. a OR abc . 18. a OR abc . 7.8.14. a OR abc . 7. 14. 18. a OR abc 1790er
    OR abc . 20?()/&!%000 a  OR abc . 2,9-3,5 a OR abc 45, 59 a OR abc . 14 z OR abc  1. e OR abc  v. 2 a
    """
    string = re.sub('(?<!\w)(\d+)([\W\s]+|)|([\W\s]+)\d+', ' ', string)  # TODO: check later
    # Alternative: ((\d+)(.|\s{1,3}|)\d+)(.|\s)(?! er)
    return string

nlp = German()
sbd = nlp.create_pipe('sentencizer')
nlp.add_pipe(sbd)

def Sentencizer(string, verbose=False):
    """
    requires from importing language from spacy and loading of sentence boundary detection:
    from spacy.lang.de import German
    nlp = German()
    sbd = nlp.create_pipe('sentencizer')
    nlp.add_pipe(sbd)

    for some single strings nlp() cannot process (rare, e.g. 'nan'), exclude those; except pass solve later
    """
    sents_list = []
    try:
        doc = nlp(string)
        for sent in doc.sents:
            sents_list.append(sent.text)
    except:
        if verbose: print("###\tSentencizer(): nlp() could not read string: '{}'".format(string))
        pass
    return sents_list

def ArticlesToLists(string):
    """
    Put Articles into a nested list in list so we can apply same fcts as we do to sentences and paragraphs
    :param string: str, article
    :return: list(str)
    """
    return [string]

def WordRemover(listOfSents, dropWords):
    """
    drop words from listOfSents which are specified in dropWords
    """
    cleanedlistOfSents = []
    for sent in listOfSents:
        temp_sent = sent
        for word in dropWords:
            temp_sent = re.sub('(?<![-/&]|\w){}(?![-/&]|\w)'.format(word), '', temp_sent)
        cleanedlistOfSents.append(temp_sent)
    return cleanedlistOfSents


def LinkRemover(listOfSents):
    """
    removes any kind of link
    """
    cleanedlistOfSents = []
    for sent in listOfSents:
        temp_sent = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            " ", sent)
        cleanedlistOfSents.append(temp_sent)
    return cleanedlistOfSents


def MailRemover(listOfSents):
    """
    removes mail addresses
    """
    cleanedlistOfSents = []
    for sent in listOfSents:
        temp_sent = re.sub(r'\S+@\S+', ' ', sent)
        cleanedlistOfSents.append(temp_sent)
    return cleanedlistOfSents


def SpecialCharCleaner(listOfSents):
    """
    cleans special characters, trash hyphens and punctuations;
    apply loop fct to each list in pandas cell
    """
    p = re.compile(r"(\b[-'/.&\\]\b)|[\W_]")
    return [p.sub(lambda m: (m.group(1) if m.group(1) else " "), x) for x in listOfSents]


nlp2 = spacy.load('de_core_news_md') #, disable=['ner', 'parser']


def POSlemmatizer(listOfSents, POStag=p['POStag']):
    """
    POS tag words in sentences, lemmatize words, remove stop words

    :param listOfSents: nested list of articles where sentences are nested
    :param POStag: str or list; e.g. 'NN', or 'NNV'
    :return: listOfSents
    """
    POStaggedlist = []
    # catch POStag
    if POStag == 'NN':
        POStaglist = ['NN']
    if POStag == 'NNV':
        POStaglist = ['NN', 'VAFIN', 'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF',
                      'VMPP', 'VVFIN', 'VVIMP', 'VVINF', 'VVIZU', 'VVPP']

    # First, POStag words and remove stop words
    for sent in listOfSents:
        sent_nlp2, sent_POStagged, sent_tokens = nlp2(sent), [], []
        for token in sent_nlp2:
            if token.tag_ in POStaglist and not token.is_stop:
                sent_POStagged.append(token)

        # Second, lemmatize words from first step
        for token in sent_POStagged:
            if token.tag_ == 'NN':
                temp_ = nlp2(token.string.title())
                for t in temp_:
                    sent_tokens.append(t.lemma_)
            else:
                temp_ = nlp2(token.string)
                for t in temp_:
                    sent_tokens.append(t.lemma_)

        # Third, make back lower again
        sent_tokens_lower = [i.lower() for i in sent_tokens]

        # Forth, add all POStagged words to final list
        POStaggedlist.append(sent_tokens_lower)

    return POStaggedlist


# Load Lemmatizer
# lemmatizer = GermaLemma()


def Lemmatization(listOfSents):
    """
    Lemmatizer of POS tagged words in sentences. Run this fct after SentencePOStagger()
    """
    lemmalist = []
    for sent in listOfSents:
        lemmalist.append([])
        sent_ = nlp2(sent)
        for token in sent:
            # token_lemma = lemmatizer.find_lemma(token.text, token.tag_)
            token_lemma = token.lemma_
            token_lemma = token_lemma.lower()
            lemmalist[-1].append(token_lemma)
    return lemmalist


def SentenceTokenizer(sent):
    """

    :param sent: 1 sentence as string (after preprocessing!)
    :return: sentence as list of tokens without punctuation
    """

    tokenizedlist = []
    #Set up tokenizer
    tokenizer = Tokenizer(nlp.vocab)
    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    tokens = tokenizer(sent)

    for token in tokens:
        # ignore punctuation
        if len(token.text) > 1:
            tokenizedlist.append(token.text)
    return tokenizedlist

#ToDo: delete Sentencelisttokenizer??
def SentenceListTokenizer(listOfSents):
    """
    load SetupTokenizer() first
    """

    # Set up tokenizer
    tokenizer = Tokenizer(nlp.vocab)
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    tokenizedlist = []
    for sent in listOfSents:
        token = tokenizer(sent)
        tokenizedlist.append(token.text)
    return tokenizedlist

def TokensCleaner(listOfSents, minwordinsent, minwordlength, drop=False):
    """
    this function filters out lists that contain too less and to short tokenized words, specified with minwordinsent,
    minwordlength, else, if drop=True, also keeps filtered out lists, but empty (needed for having same dimension
    when making long)
    Run after POStagger() and after Lemmatization(), final step before exporting.

    :param listOfSents: list, elements are sentences
    :param minwordinsent: int, min word that should be contained in a sentence
    :param minwordlength: int, min length of words that should be contained in a sentence
    :param drop: True, False, if True, drop empty lists
    :return: list if tokenized words with specified minwordinsent, minwordlength
    """
    cleanedlistOfSents = []
    for sent in listOfSents:
        filteredSent = []
        # first check whether 'nan' included
        tempsent = [x for x in sent if x!='nan']
        # Only append if e.g. list-length<3 and word-length<3
        for word in tempsent:
            # filter out short words and short sentences
            if (len(tempsent) >= minwordinsent) and (len(word) >= minwordlength):
                filteredSent.append(word)
        # append filtered sentence list to cleaned list of sentences only if it still contained 3 words or more
        if (len(filteredSent) >= minwordinsent):
            cleanedlistOfSents.append(filteredSent)
        elif not drop:
                cleanedlistOfSents.append([])
    return cleanedlistOfSents


def NormalizeWords(string):
    """
    Normalize Words (Preserve words by replacing to synonyms and write full words instead abbrev.)
    """

    # Normalize e-mobility related words
    string = string.replace('co2', 'kohlenstoffdioxid').replace('co²', 'kohlenstoffdioxid').replace('co 2',
                                                                                                    'kohlenstoffdioxid')
    string = string.replace('km/h', 'kilometerprostunde').replace('km-h', 'kilometerprostunde').replace('km /h',
                                                                                                        'kilometerprostunde')
    string = string.replace('g/km', 'grammprokilometer').replace('g-km', 'grammprokilometer').replace('g /km',
                                                                                                      'grammprokilometer')
    string = string.replace('g/cm³', 'grammprokubikmeter').replace('cm³', 'kubikmeter')
    string = string.replace('/km', 'prokilometer').replace('km', 'kilometer')
    string = string.replace('m-s', 'meterprosekunde').replace('m/s', 'meterprosekunde')
    string = string.replace('mio.', 'million').replace('mrd.', 'milliarde').replace('mill.', 'million')
    string = string.replace('kwh', 'kilowattstunde').replace('mwh', 'megawattstunde').replace('kw/h', 'kilowattstunde')
    string = string.replace('kw/', 'kilowatt').replace(' kw ', ' kilowatt ').replace('-kw-', 'kilowatt')
    string = string.replace('mw/h', 'megawattstunde').replace('kw-h', 'kilowattstunde').replace('mw-h',
                                                                                                'megawattstunde')
    string = string.replace('v-12', 'vzwölf').replace('v12', 'vzwölf').replace('v.12', 'vzwölf').replace(' v 12 ',
                                                                                                         ' vzwölf ')
    string = string.replace('v-10', 'vzehn').replace('v8', 'vzehn').replace('v.10', 'vzehn').replace(' v 10 ',
                                                                                                     ' vzehn ')
    string = string.replace('v-8', 'vacht').replace('v8', 'vacht').replace('v.8', 'vacht').replace(' v 8 ', ' vacht ')
    string = string.replace('v-6', 'vsechs').replace('v6', 'vsechs').replace('v.6', 'vsechs').replace(' v 6 ',
                                                                                                      ' vsechs ')
    string = string.replace('f&e', 'fue').replace(' e 10 ', ' ezehn ').replace(' e10 ', ' ezehn ')
    string = string.replace('formel 1', 'formeleins').replace('formel1', 'formeleins').replace('formel-1', 'formeleins')
    string = string.replace(' ps ', ' pferdestärke ').replace(' ps', ' pferdestärke')
    string = string.replace(' kg ', ' kilogramm ').replace(' g ', ' gramm ')
    string = string.replace('-v-', '-volt-').replace(' v ', ' volt ')
    string = string.replace(' nm ', ' newtonmeter ').replace(' m ', ' meter ')
    string = string.replace(' h ', ' stunden ').replace(' h.', ' stunden.')

    # car models
    string = string.replace('i3', 'idrei').replace('i10', 'izehn').replace('e3', 'edrei').replace(' e 3 ',
                                                                                                  ' edrei ').replace(
        'i8', 'iacht')
    string = string.replace('s base', 'sbase').replace('ev1', 'eveins')
    string = string.replace('urban ev', 'urbanev')
    string = string.replace('vw up', 'vwup').replace('benz eq', 'benzeq')
    string = string.replace('leaf e', 'leafe').replace('leaf e+', 'leafeplus')
    string = string.replace('soul ev', 'soulev')
    string = string.replace('i.d.', 'vwid').replace(' id.', ' vwid').replace('vw id. ', 'vwid ').replace(' id. ',
                                                                                                         ' id ')
    string = string.replace('vw id.3', 'vwiddrei').replace('id.3', 'vwiddrei')
    string = string.replace('vwid neo', 'vwidneo')
    string = string.replace('mini e ', 'minie ').replace('mini e. ', 'minie ')
    string = string.replace('fluence z.e.', 'fluenceze').replace('fluence z.e.', 'fluenceze').replace('fluence ze ',
                                                                                                      'fluenceze ').replace(
        'fluence ze. ', 'fluenceze ')
    string = string.replace('kangoo z.e ', 'kangooze ').replace('kangoo z.e.', 'kangooze').replace('kangoo ze ',
                                                                                                   'kangooze ').replace(
        'kangoo ze. ', 'kangooze ')
    string = string.replace('s60', 'ssechzig').replace('d70', 'dsiebzig').replace('70d', 'siebzigd').replace('s 70d',
                                                                                                             'ssiebzigd')
    string = string.replace('e.go life', 'e.golife').replace('s85', 'sfünfundachtzig')

    # names, companies, terms
    string = string.replace(' vw', ' volkswagen')
    string = string.replace('ig metall', 'igmetall').replace('ig-metall', 'igmetall')
    string = string.replace('z.e.', 'zeroemission')

    # titel
    string = string.replace('dr.', 'doktor').replace('prof.', 'professor').replace('phd.', 'doktor').replace(' phd ',
                                                                                                             'doktor')
    string = string.replace('dipl.-ing.', 'diplomingenieur').replace('dipl-ing.', 'diplomingenieur')
    string = string.replace('b.a.', 'bachelor').replace('b.sc.', 'bachelor').replace('ll.b.', 'bachelor')
    string = string.replace('m.a.', 'master').replace('m.sc.', 'master').replace('ll.m.', 'master')
    string = string.replace('lic.', 'licentiatus').replace('rer.', 'rerum').replace('publ.', 'publicarum').replace(
        ' reg.', ' regionalum')
    string = string.replace('mag.', 'magister').replace('iur.', 'iuris').replace('dipl.-inf.', 'diplominformatiker')
    string = string.replace('dipl.-betriebsw.', 'diplombetriebswirt').replace('-inf.', 'informatiker').replace('päd.',
                                                                                                               'pädagoge')
    string = string.replace('dipl.-inform.', 'diplominformatiker').replace('-wirt', 'wirt').replace('dipl.', 'diplom')
    string = string.replace('kfm.', 'kaufmann').replace('kffr.', 'kauffrau').replace('psych.', 'psychologe')
    string = string.replace('techn.', 'technik').replace('verw.', 'verwaltung').replace('betriebsw.', 'betriebswirt')
    string = string.replace('volksw.', 'volkswirt').replace('jur.', 'jurist').replace('phil.', 'philosophiae')

    # normalize mostly used abbrev.
    string = string.replace(' st. ', ' sankt ')
    string = string.replace('abb.', 'abbildung').replace('abs.', 'absatz').replace('abschn.', 'abschnitt')
    string = string.replace('anl.', 'anlage').replace('anm.', 'anmerkung').replace('art.', 'artikel').replace('aufl.',
                                                                                                              'auflage')
    string = string.replace('bd.', 'band').replace('bsp.', 'beispiel').replace('bspw.', 'beispielsweise')
    string = string.replace('bzgl.', 'bezüglich').replace('bzw.', 'beziehungsweise').replace('bt-drs.',
                                                                                             'bundestragsdrucksache')
    string = string.replace('beschl.v.', 'beschluss von').replace('beschl. v.', 'beschluss von')
    string = string.replace('ca.', 'circa').replace('d.h.', 'dasheißt').replace('ders.', 'derselbe')
    string = string.replace('dgl.', 'dergleichen').replace('dt.', 'deutsch')
    string = string.replace('e.v.', 'eingetragenerverein').replace('etc.', 'etcetera').replace('evtl.', 'eventuell')
    string = string.replace(' f.', ' fortfolgend').replace(' ff.', ' fortfolgend').replace('gem.', 'gemäß')
    string = string.replace('ggf.', 'gegebenenfalls').replace('grds.', 'grundsätzlich')
    string = string.replace('hrsg.', 'herausgeber').replace('i.a.', 'imauftrag').replace('i.d.f.', 'in der fassung')
    string = string.replace('i.d.r.', 'in der regel')
    string = string.replace('i.d.s.', 'in diesem sinne').replace('i.e.', 'im ergebnis').replace('i.v.', 'in vertretung')
    string = string.replace('i. d. s.', 'in diesem sinne').replace('i. e.', 'im ergebnis').replace('i. v.',
                                                                                                   'in vertretung')
    string = string.replace('i.v.m.', 'in verbindung mit')
    string = string.replace('i.ü.', 'im übrigen').replace('inkl.', 'inklusive').replace('insb.', 'insbesondere')
    string = string.replace('i. ü.', 'im übrigen').replace('mwst.', 'mehrwertsteuer')
    string = string.replace('m.e.', 'meines erachtens').replace('max.', 'maximal').replace('min.', 'minimal')
    string = string.replace('n.n.', 'nomennescio').replace('nr.', 'nummer').replace('o.a.', 'oben angegeben')
    string = string.replace('o.ä.', 'oder ähnliches').replace('o.g.', 'oben genannt')
    string = string.replace('o. ä.', 'oder ähnliches').replace('o. g.', 'oben genannt')
    string = string.replace('p.a.', 'proanno').replace('pos.', 'position').replace('pp.', 'perprocura')
    string = string.replace('rs.', 'rechtssache').replace('rspr.', 'rechtsprechung').replace('sog.', 'sogenannt')
    string = string.replace('s.a.', 'siehe auch').replace('s.o.', 'siehe oben').replace('s.u.', 'siehe unten')
    string = string.replace('s. a.', 'siehe auch').replace('s. o.', 'siehe oben').replace('s. u.', 'siehe unten')
    string = string.replace('tab.', 'tabelle').replace('tel.', 'telefon').replace('tsd.', 'tausend')
    string = string.replace('u.a.', 'unter anderem').replace('u.ä.', 'und ähnliches').replace('u.a.m.',
                                                                                              'und anderes mehr')
    string = string.replace('u.a ', 'unter anderem ').replace('u.ä ', 'und ähnliches ').replace('u.a.m ',
                                                                                                'und anderes mehr ')
    string = string.replace('u. a.', 'unter anderem').replace('u. ä.', 'und ähnliches').replace('u. a. m.',
                                                                                                'und anderes mehr')
    string = string.replace('u. u.', 'unter umständen').replace('urt. v.', 'urteil vom').replace('urt.v.', 'urteil von')
    string = string.replace('usw.', 'und so weiter').replace('u.v.m.', 'und vieles mehr')
    string = string.replace('usw.', 'und so weiter').replace('u. v. m.', 'und vieles mehr')
    string = string.replace('v.a.', 'vor allem').replace('v.h.', 'vom hundert').replace('vgl.', 'vergleiche')
    string = string.replace('v. a.', 'vor allem').replace('vgl.', 'vergleiche')
    string = string.replace('vorb.', 'vorbemerkung').replace('vs.', 'versus')
    string = string.replace('z.b.', 'zum beispiel').replace('z.t.', 'zum teil').replace('zz.', 'zurzeit')
    string = string.replace('z. b.', 'zum beispiel').replace('z. t.', 'zum teil')
    string = string.replace('k. a.', 'keine angabe').replace('k.a.', 'keine angabe')
    string = string.replace('zzt.', 'zurzeit').replace('ziff.', 'ziffer').replace('zit.', 'zitiert').replace('zzgl.',
                                                                                                             'zuzüglich')

    # other
    string = string.replace('vdi nachrichten', ' ').replace('siehe grafik', ' ').replace(' nan ', ' ')

    # put last rules here which are not affected by above ones
    string = string.replace('%', 'prozent').replace('€', 'euro').replace('$', 'dollar')
    string = string.replace(' s ', ' sekunden ').replace(' kw', ' kilowatt').replace('°c', 'gradcelsius')

    # correct false Lemmatization of Spacy
    string = string.replace('lithium-ionen-batterien', 'lithium-ionen-batterie')
    string = string.replace('lithium-ionen-akkus', 'lithium-ionen-akku')
    string = string.replace('batteriezellen', 'batteriezelle')
    string = string.replace('ladestationen', 'ladestation').replace('ladesäulen', 'ladesäule')
    string = string.replace('ladepunkte', 'ladepunkt')
    string = string.replace('e-mobilität', 'elektromobilität')
    string = string.replace('e-mobil', 'elektromobil').replace('e-mobile', 'elektromobil')
    string = string.replace('arbeitsplätze', 'arbeitsplatz')
    string = string.replace('e-autos', 'elektroauto').replace('e-auto', 'elektroauto')
    string = string.replace('elektro-autos', 'elektroauto').replace('elektro-auto', 'elektroauto')
    string = string.replace('elektrofahrzeuge', 'elektrofahrzeug').replace('elektroautos', 'elektroauto')
    string = string.replace('elektroräder', 'elektrorad').replace('elektro-fahrräder', 'elektro-fahrrad')
    string = string.replace('e-bikes', 'e-bike')
    string = string.replace('e-antrieb', 'elektroantrieb').replace('elektromotoren', 'elektromotor')
    string = string.replace('e-smart', 'elektro-smart').replace('pedelecs', 'pedelec')
    string = string.replace('e-busse', 'elektrobus').replace('e-bus', 'elektrobus').replace('elektrobusse',
                                                                                            'elektrobus')
    string = string.replace('elektrobuss', 'elektrobus').replace('bürgermeisterin', 'bürgermeister')
    string = string.replace('elektroauto-batterien', 'elektroauto-batterie')
    string = string.replace('konzeptfahrzeuge', 'konzeptfahrzeug').replace('festkörperbatterien', 'festkörperbatterie')
    string = string.replace('haushaltsgeräte', 'haushaltsgerät').replace('e-taxis', 'e-taxi')

    string = string.replace(' analysten', ' analyst').replace(' bestückt', ' bestücken')


    string = string.replace('obusse', 'obus').replace('forschungsministerin', 'forschungsminister')
    string = string.replace('stromtankstellen', 'stromtankstelle').replace(' module', ' modul')
    string = string.replace(' superkondensatoren', ' superkondensator')
    string = string.replace('vorstandsvorsitzende', 'vorstandsvorsitzender')
    string = string.replace('installateure', 'installateur').replace('tesla-batterien', 'tesla-batterie')
    string = string.replace(' batteriemodulen', ' batteriemodule').replace('batterie-packs', 'batterie-pack')
    string = string.replace('speicherseen', 'speichersee').replace('serienautos', 'serienauto')
    string = string.replace(' lkw', ' lastwagen').replace('anoden', 'anode').replace(' auslieferungen', ' auslieferung')
    string = string.replace(' boni', ' bonus').replace(' e-fuels', ' e-fuel').replace(' kaufprämien', ' kaufprämie')
    string = string.replace('wechselstationen', 'wechselstation').replace('bleiakkus', ' bleiakkus')
    string = string.replace('solarzellen', 'solarzelle')
    string = string.replace('privathaushalte', 'privathaushalt').replace(' zellen', ' zelle').replace(' trucks',
                                                                                                      ' truck')
    string = string.replace('karbonfasern', 'karbonfaser').replace('kohlenstoffdioxid-emissionen',
                                                                   'kohlenstoffdioxid-emission')
    string = string.replace('kilometern', 'kilometer')
    string = string.replace('lässt', 'lassen')

    # delete
    string = string.replace(' sion ', ' ').replace(' dinslaken', ' ').replace(' bahnen', ' '). replace(' preisen', ' ')
    string = string.replace(' modeln', ' ').replace(' werken', ' ').replace(' golfen', ' ').replace(' frau', ' ')
    string = string.replace(' gasen', ' ').replace(' rädern', ' ').replace(' mann', ' ')
    string = string.replace(' winden', ' ').replace(' dingen', ' ').replace(' mark', ' ').replace(' hausen', ' ')
    string = string.replace(' bild', ' ').replace(' laeden', ' ').replace(' münchner', ' ').replace(' bauchen', ' ')
    string = string.replace(' datum', ' ').replace(' frage', ' ').replace(' achatz', ' ')
    string = string.replace(' n-ergie', ' ').replace(' elektro-smartikel', ' ').replace(' strombetriebenen', ' ')
    string = string.replace(' ene ', ' ').replace(' kolbe ', ' ').replace(' topfen', ' ')
    string = string.replace(' herstellagier', ' ').replace(' dachen', ' ').replace(' löer', ' ')
    string = string.replace(' montag', ' ').replace(' dienstag', ' ').replace(' mittwoch', ' ')
    string = string.replace(' donnerstag', ' ').replace(' freitag', ' ')
    string = string.replace(' januar', ' ').replace(' februar', ' ').replace(' märz', ' ').replace(' april', ' ')
    string = string.replace(' mai', ' ').replace(' juni', ' ').replace(' juli', ' ').replace(' august', ' ')
    string = string.replace(' september', ' ').replace(' oktober', ' ').replace(' november', ' ').replace(' dezember',
                                                                                                          ' ')

    return string


def ParagraphSplitter(listOfPars, splitAt):
    """
    Remove text which defines end of articles;
    Strings = 'graphic', 'foto: classification language', 'classification language', 'kommentar seite '
    """
    splitPars, splitHere = [], False
    for par in listOfPars:
        for splitstring in splitAt:
            if splitstring in par:
                splitHere = True
        if not splitHere:
            splitPars.append(par)
        else:
            break
    return splitPars


def ProcessforSentiment(listOfSents):
    """
    Process sentences before running Sentiment Analysis, replace ;: KON by , and drop .!? and lemmatize
    :param listOfSents: list of sentences where sentences are str
    :return: listOfSentenceparts
        [['sentencepart1', 'sentencepart2', ...], [], [], ...]
        which are split by ,
    """

    temp_article, processed_article, final_article = [], [], []
    for sent in listOfSents:
        # First drop .?! and brackets
        temp_sent = sent.replace('.', '').replace('!', '').replace('?', '').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        # Replace :; by ,
        temp_sent = temp_sent.replace(';', ',').replace(':', ',')
        # apply nlp2 to temp_sent
        temp_sent_nlp = nlp2(temp_sent)
        # process each token and 'translate' konjunction or ;: to ,
        temp_sent = []
        for token in temp_sent_nlp:
            if token.tag_=='KON':
                temp_sent.append(',')
            else:
                temp_sent.append(token.text)

        # put all tokens to a string (but split later by normalized ,)
        sent_string = ' '.join(temp_sent)

        # prepare for lemmatization
        sent_string = nlp2(sent_string)

        # Second, loop over all tokens in sentence and lemmatize them
        sent_tokens = []
        for token in sent_string:
            sent_tokens.append(token.lemma_)
        processed_article.append(sent_tokens)

    # Put together tokenized, lemmatized elements of lists to a string
    processed_article = [' '.join(i) for i in processed_article]
    # Split by normalized commas
    for sent in processed_article:
        final_article.append(sent.split(','))

    # Flatten list
    final_article = FlattenList(final_article)

    # strip strings
    final_article = [x.strip() for x in final_article]

    # drop empty elements
    final_article = [x for x in final_article if x != '']
    # final_article = [x.strip() for x in final_article if x.strip()]

    #  return a string with lemmatized words and united sentence splits to ,
    return final_article

def IgnoreWarnings():
    # ignore by messages when extracting sentiment scores
    warnings.filterwarnings("ignore", message="Mean of empty slice.")
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
    # warnings.filterwarnings("ignore", category=FutureWarning)
    return print('Ignoring warnings when extracting sentiment scores')

def CapitalizeNouns(doc):
    """
    help function for Spacy when lemmatizing, make nouns capitalized
    :param doc: str, doc
    :return: str, doc but with capitalized nouns
    """
    # # initialize doc in nlp2()
    # doc_temp, doc_capitalized = nlp2(doc), []
    # # loop over all elements and capitalize if tag is 'NN'
    # for token in doc_temp:
    #     if token.tag_ == 'NN':
    #         doc_capitalized.append(token.string.title())
    #     else:
    #         doc_capitalized.append(token.string)
    # return ' '.join(doc_capitalized)
    doc_cap = []
    for d in doc:
        tagged_sent = [(w.text, w.tag_) for w in nlp2(d)]
        normalized_sent = [w.capitalize() if t in ["NN", "NNS", 'NE'] else w for (w, t) in tagged_sent]
        try:
            normalized_sent[0] = normalized_sent[0].capitalize()
        except IndexError:
            normalized_sent = ''
        doc_cap.append(' '.join(normalized_sent))
    return doc_cap
