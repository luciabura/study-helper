from spacy.tokens import doc

from question_generation.question_provider import sequence_surprize
from utilities import NLP


class Question:
    def __init__(self, question, sentence, answer):
        self.content = NLP(question)
        # Assumes we get a sentence object of the form described by Sentence in sentence_provider
        self.sentence = sentence
        self.key_phrases = sentence.key_phrases
        self.answer = answer

        self.score = 0

        self.__compute_score()

    def __compute_score(self):
        # Exclude questions which are referring expressions
        # eg: What is he looking at?

        # Filter out questions which don't make sense
        for tok in self.content:
            if (tok.tag_ == 'PRON' or tok.tag_ == 'PRP') or \
                    (tok.tag_ == 'DT' and tok.dep_.startswith('nsubj')):
                if tok.text in ['we', 'you']:
                    continue
                self.score = 0
                return

        # Calculating the score for keywords within Question
        score_q = 0
        div = 0
        alpha = 0.955

        for kp in self.sentence.key_phrases:
            if kp.text in self.content.text.lower():
                div += 1
                surprise_factor_normalized = sequence_surprize(kp.text)
                score_q += alpha * kp.score + (1 - alpha) * surprise_factor_normalized
        if div:
            score_q = score_q / div

        # Calculating the score for keywords within answer phrase
        text_answer = [tok.lower_ for tok in self.answer]
        text_answer = ' '.join(text_answer)

        score_kp = 0
        div = 0
        for kp in self.sentence.key_phrases:
            if kp.text in text_answer:
                div += 1
                surprise_factor_normalized = sequence_surprize(kp.text)
                score_kp += alpha * kp.score + (1 - alpha) * surprise_factor_normalized

        if div:
            score_kp = score_kp / div

        beta = 0.2
        if score_q and score_kp:
            score = (1.0 + beta ** 2) * (score_q * score_kp) / (score_q + (beta ** 2) * score_kp)
        elif score_q:
            score = score_q
        else:
            score = score_kp * (1 - beta)

        # if len(self.content) > 0:
        #     score /= math.log(len(self.content))

        self.score = score

    def get_similarity(self, question2):
        if isinstance(question2, Question):
            return self.content.similarity(question2.content)
        elif isinstance(question2, str):
            return self.content.similarity(NLP(question2))
        elif isinstance(question2, doc):
            return self.content.similarity(question2)
        else:
            return None