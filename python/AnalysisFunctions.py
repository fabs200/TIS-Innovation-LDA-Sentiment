# Help functions for LDA and Sentiment Analysis

def MakeCandidates(sent, verbose=False, get='candidates'):
    """
    prepares a nested list of candidates, make sure df_sepl['phrase'] is loaded
    :param sent: input is a full sentence as string
    :param verbose: display
    :param get: default 'candidates', else specify 'negation' to geht same list in lists but only negation words
    :return: nested list of lists where each nested list is separated by the POS tag $,
    """
    if get=='candidates':
        candidates = []
        for token in sent:
            if verbose: print(token.text, token.tag_)
            if token.tag_.startswith(('NN', 'V', 'ADV', 'ADJ', '$,')):
                if df_sepl['phrase'].str.contains(r'(?:\s|^){}(?:\s|$)'.format(token)).any() or token.tag_.startswith(('$,')):
                    candidates.append(token.text)
        if verbose: print(candidates)

        # get index of elements at which we want to split the list
        split_at_index = [i for i, j in enumerate(candidates) if j == ',']

        # Split list at index and make nested lists
        candidates = [candidates[i : j] for i, j in zip([0] + split_at_index, split_at_index + [None])]
        if verbose: print(candidates)

        # Drop elements with comma
        temp_candidates, candidates = candidates, []
        for li in temp_candidates:
            temp_li = []
            for el in li:
                if el is not ',':
                    print(el)
                    temp_li.append(el)
                print(temp_li)
            candidates.append(temp_li)

        # Drop empty lists
        candidates = [x for x in candidates if x]
        if verbose: print(candidates)

    if get=='negation':
        # TODO
    return candidates

