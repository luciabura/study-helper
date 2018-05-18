class Keyword(object):

    def __init__(self, token, score, sentence):
        self.token = token
        self.text = self.token.lower_
        self.score = score
        self.sentence = sentence
        self.doc = token.doc


class KeyPhrase(object):

    def __init__(self, start_index, end_index, sentence, keywords):
        self.span = sentence[start_index:end_index + 1]
        self.length = len(self.span)
        self.text = self.span.string.strip().lower()
        self.keywords = keywords
        self.sentence = sentence
        self.score = self.__calculate_score()

    def __calculate_score(self):
        score = 0
        for key in self.keywords:
            score += key.score
        return score

    def similarity(self, token):
        return token.similarity(self.span)