import glob
import os
from spacy.attrs import *
from spacy.matcher import Matcher

from data import DATA_DIR
from question_generation import ANY_TOKEN
from text_processing.preprocessing import clean_to_doc
from utilities import NLP
from utilities.read_write import read_file

SUMM_DIR = os.path.join(DATA_DIR, 'summarization_eval')
SUMM_BODY = os.path.join(SUMM_DIR, 'body')

MATCHER = Matcher(NLP.vocab)


def initialize_patetrns():
    acomp_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'acomp'}]
    acomp_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'acomp'}]

    sv = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}]
    sv_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}]

    attr_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'attr'}]
    attr_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'attr'}]

    ccomp_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'ccomp'}]
    ccomp_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'ccomp'}]

    direct_object_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'}]
    direct_object_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'}]

    prep_object_1 = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'prep'},
                     ANY_TOKEN, {DEP: 'pobj'}]
    prep_object_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'prep'},
                     ANY_TOKEN, {DEP: 'pobj'}]

    d_obj_p_obj = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'},
                   ANY_TOKEN, {DEP: 'pobj'}]
    d_obj_p_obj_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dobj'},
                     ANY_TOKEN, {DEP: 'pobj'}]
    iobj_dobj = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'iobj'},
                   ANY_TOKEN, {DEP: 'dobj'}]
    iobj_dobj_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'iobj'},
                     ANY_TOKEN, {DEP: 'dobj'}]
    d_obj_p_obj_inv = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'pobj'},
                       ANY_TOKEN, {DEP: 'dobj'}]
    d_obj_p_obj_inv_2 = [{DEP: 'nsubjpass'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'pobj'},
                         ANY_TOKEN, {DEP: 'dobj'}]
    dative_pobj = [{DEP: 'nsubj'}, ANY_TOKEN, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'dative'},
                       ANY_TOKEN, {DEP: 'pobj'}]

    agent_1 = [{DEP: 'nsubj'}, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'agent'}, ANY_TOKEN, {DEP: 'pobj'}]
    agent_2 = [{DEP: 'nsubjpass'}, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'agent'}, ANY_TOKEN, {DEP: 'pobj'}]

    # MATCHER.add("ATTR", None, attr_1)
    # MATCHER.add("ATTR", None, attr_2)
    # 14.800945999211667

    # MATCHER.add("ACOMP", None, acomp_1)
    # MATCHER.add("ACOMP", None, acomp_2)
    # 9.893575088687426

    # MATCHER.add("CCOMP", None, ccomp_1)
    # MATCHER.add("CCOMP", None, ccomp_2)
    # 12.204258675078865
    #
    MATCHER.add("SV", None, sv)
    MATCHER.add("SV", None, sv_2)

    agent_1 = [{DEP: 'nsubj'}, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'agent'}, ANY_TOKEN, {DEP: 'pobj'}]
    agent_2 = [{DEP: 'nsubjpass'}, {POS: 'VERB', DEP: 'ROOT'}, ANY_TOKEN, {DEP: 'agent'}, ANY_TOKEN, {DEP: 'pobj'}]

    MATCHER.add("AGENT", None, sv)
    MATCHER.add("AGENT", None, sv_2)

    # MATCHER.add("DOBJ", None, direct_object_1)
    # MATCHER.add("DOBJ", None, direct_object_2)
    # 55.87539432176656 -- need to draw down dobj-pobj

    # MATCHER.add("POBJ", None, prep_object_1)
    # MATCHER.add("POBJ", None, prep_object_2)
    # 77.64195583596214

    # MATCHER.add("AGENT", None, agent_1)
    # MATCHER.add("AGENT", None, agent_2)
    # 2.0110410094637223

    # MATCHER.add("DOBJ-POBJ", None, d_obj_p_obj)
    # MATCHER.add("DOBJ-POBJ", None, d_obj_p_obj_2)
    # MATCHER.add("DOBJ-POBJ", None, d_obj_p_obj_inv)
    # MATCHER.add("DOBJ-POBJ", None, d_obj_p_obj_inv_2)
    # 48.20583596214511 -- all
    # only last 2 -- 25.019716088328074

    # MATCHER.add("XCOMP", xcomp_1)
    # MATCHER.add("XCOMP", xcomp_2)


def count_matches():
    os.chdir(SUMM_BODY)
    total_count = 0
    total_matches = 0
    for file in glob.glob("*body.txt"):

        input_path = os.path.join(SUMM_BODY, file)
        file_text = read_file(input_path)
        doc = clean_to_doc(file_text)
        sentences = doc.sents
        for sentence in sentences:
            total_count += 1
            sent_doc = sentence.as_doc()
            matches = MATCHER(sent_doc)

            if matches:
                total_matches+=1
            # for ent_id, start, end in matches:
            #     pattern_name = NLP.vocab.strings[ent_id]
                # handle_name(pattern_name)

    percentage = (total_matches*1.0/total_count)*100
    print(percentage)


def inspect_text_pattern_distribution(text):
    """gets percentage for 5-6 patterns for a particular text"""
    pass


def aggregate_pattern_percentage(folder_path):
    """Look into a folder and appply inspect all texts for patterns
    Proceed to """
    pass


if __name__ == '__main__':
    initialize_patetrns()
    count_matches()