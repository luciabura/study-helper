"""
Takes qq system output
Heilman QG output

Measures:
1) syntactical correctness
2) Similarity with gold questions -> spacy similarity + words present + phrase sequence
3*) relevance to topic if topic exists
"""


from sklearn.decomposition import LatentDirichletAllocation
from text_processing.preprocessing import spacy_word_tokenize
from utilities.read_write import read_file
if __name__ == '__main__':
    text = "What is the colour of his hair?"
    text2 = "What are John doing?"
    text3 = "What is John doing?"
    lda = LatentDirichletAllocation()
    t_tokens = spacy_word_tokenize(text)

    corpus_filepath = input("Corpus filepath:")
    corpus_text = read_file(corpus_filepath)
    tokens = spacy_word_tokenize(corpus_text)

    lda.fit(tokens)
    p_s = lda.perplexity(t_tokens)
    # p_s2 = LatentDirichletAllocation.perplexity(text2)
    # p_s3 = LatentDirichletAllocation.perplexity(text3)
    """Our perplexity for the second question should be higher"""
    print(p_s)
    # print(p_s2)
    # print(p_s3)

def similarity_overlap_score(q1, q2):
    """Takes two questions as text"""
    pass


def semantic_overlap_score(s1, s2):
    pass
