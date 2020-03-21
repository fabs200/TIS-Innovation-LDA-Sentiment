from python.AnalysisFunctions import Load_SePL, MakeCandidates, GetSentiments
# Dissertation Rill (2016): S.66-73, 110-124

# Load SePL data frame
df_sepl = Load_SePL()

sentence1 = 'anfangs war er mir sehr unsympathisch' #included in SePL: anfangs sehr unsympathisch
sentence2 = 'ich finde ihn ganz gut, aber ich bin nicht sehr begeistert von seinem auto' #negation not included in SePL
sentence3 = 'ich habe heute, am 19.03.2020 Geburtstag, keine geile Party'
sentence4 = 'Er kann nicht Auto fahren, er weiß nichts über usa, er traut sich nicht hinzufliegen, er hat kein toilettenpapier, er ist einfach zu schlecht und mies'

"""
first step: identification of suitable candidates for opinionated phrases suitable candidates: nouns, adjectives, 
adverbs and verbs
"""

candidates = MakeCandidates(sentence4, df_sepl, get='candidates')
negation_candidates = MakeCandidates(sentence4, df_sepl, get='negation')

print('candidates:', candidates)
print('negation:', negation_candidates)

"""
second step: extraction of possible opinion-bearing phrases from a candidate starting from a candidate, 
check all left and right neighbours to extract possible phrases. The search is terminated on a comma (POS tag $,), 
a punctuation terminating a sentence (POS tag $.), a conjunction (POS-Tag KON) or an opinion-bearing word that is 
already tagged. (Max distance determined by sentence lenght)
If one of the adjacent words is included in the SePL, together with the previously extracted phrase, it is added to 
the phrase.
"""

sentimentscores, pos_tagged_words = GetSentiments(candidates, df_sepl)

print('sentimentscores:', sentimentscores)
print('pos_tagged_words:', pos_tagged_words)

"""
third step: compare extracted phrases with SePL After all phrases have been extracted, they are compared with the 
entries in the SePL. (everything lemmatized!) If no  match is found, the extracted Phrase is shortened by the last 
added element and compared again with the SePL. This is repeated until a match is found.
"""

#####(delete later): Die extrahierte Phrase „ganz sehr begeistert“ ist so nicht in der SePL enthalten. Jedoch ist die um ein Wort kürzere Phrase „sehr begeistert“ enthalten.


# SePL does not include a negated phrase for every word in the list (p.72 f.)
