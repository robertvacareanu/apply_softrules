from src.dataprocessing.semeval.dataset_converter import convert_semeval_dict
from src.dataprocessing.tacred.dataset_converter import convert_tacred_dict
from src.dataprocessing.fewrel.dataset_converter import convert_fewrel_dict
import unittest

class TestDatasetConverter(unittest.TestCase):

    def test_convert_semeval_dict(self):
        data1 = {'sentence': 'The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>.', 'relation': 3}
        data2 = {'sentence': 'The system as described above has its greatest application in an <e1>arrayed configuration</e1> of antenna <e2>elements</e2>.', 'relation': 3}
        data3 = {'sentence': 'The system as described above has its greatest application in an arrayed <e1>configuration</e1> of <e2>antenna elements</e2>.', 'relation': 3}
        data4 = {'sentence': 'The system as described above has its greatest application in an <e1>arrayed configuration</e1> of <e2>antenna elements</e2>.', 'relation': 3}
        data5 = {'sentence': 'The system as described above has its greatest application in an arrayed <e2>configuration</e2> of antenna <e1>elements</e1>.', 'relation': 3}
        data6 = {'sentence': 'The system as described above has its greatest application in an <e2>arrayed configuration</e2> of antenna <e1>elements</e1>.', 'relation': 3}
        data7 = {'sentence': 'The system as described above has its greatest application in an arrayed <e2>configuration</e2> of <e1>antenna elements</e1>.', 'relation': 3}
        data8 = {'sentence': 'The system as described above has its greatest application in an <e2>arrayed configuration</e2> of <e1>antenna elements</e1>.', 'relation': 3}

        data1_processed = convert_semeval_dict(data1)
        data2_processed = convert_semeval_dict(data2)
        data3_processed = convert_semeval_dict(data3)
        data4_processed = convert_semeval_dict(data4)
        data5_processed = convert_semeval_dict(data5)
        data6_processed = convert_semeval_dict(data6)
        data7_processed = convert_semeval_dict(data7)
        data8_processed = convert_semeval_dict(data8)

        self.assertEqual(' '.join(data1_processed['e1']), 'configuration')
        self.assertEqual(' '.join(data1_processed['e2']), 'elements')

        self.assertEqual(' '.join(data2_processed['e1']), 'arrayed configuration')
        self.assertEqual(' '.join(data2_processed['e2']), 'elements')

        self.assertEqual(' '.join(data3_processed['e1']), 'configuration')
        self.assertEqual(' '.join(data3_processed['e2']), 'antenna elements')

        self.assertEqual(' '.join(data4_processed['e1']), 'arrayed configuration')
        self.assertEqual(' '.join(data4_processed['e2']), 'antenna elements')

        self.assertEqual(' '.join(data5_processed['e1']), 'elements')
        self.assertEqual(' '.join(data5_processed['e2']), 'configuration')

        self.assertEqual(' '.join(data6_processed['e1']), 'elements')
        self.assertEqual(' '.join(data6_processed['e2']), 'arrayed configuration')

        self.assertEqual(' '.join(data7_processed['e1']), 'antenna elements')
        self.assertEqual(' '.join(data7_processed['e2']), 'configuration')

        self.assertEqual(' '.join(data8_processed['e1']), 'antenna elements')
        self.assertEqual(' '.join(data8_processed['e2']), 'arrayed configuration')


    def test_convert_tacred_dict(self):
        data1 = {'id': '61b3a5c8c9a882dcfcd2', 'docid': 'AFP_ENG_20070218.0019.LDC2009T13', 'relation': 'org:founded_by', 'token': ['Tom', 'Thabane', 'resigned', 'in', 'October', 'last', 'year', 'to', 'form', 'the', 'All', 'Basotho', 'Convention', '-LRB-', 'ABC', '-RRB-', ',', 'crossing', 'the', 'floor', 'with', '17', 'members', 'of', 'parliament', ',', 'causing', 'constitutional', 'monarch', 'King', 'Letsie', 'III', 'to', 'dissolve', 'parliament', 'and', 'call', 'the', 'snap', 'election', '.'], 'subj_start': 10, 'subj_end': 12, 'obj_start': 0, 'obj_end': 1, 'subj_type': 'ORGANIZATION', 'obj_type': 'PERSON', 'stanford_pos': ['NNP', 'NNP', 'VBD', 'IN', 'NNP', 'JJ', 'NN', 'TO', 'VB', 'DT', 'DT', 'NNP', 'NNP', '-LRB-', 'NNP', '-RRB-', ',', 'VBG', 'DT', 'NN', 'IN', 'CD', 'NNS', 'IN', 'NN', ',', 'VBG', 'JJ', 'NN', 'NNP', 'NNP', 'NNP', 'TO', 'VB', 'NN', 'CC', 'VB', 'DT', 'NN', 'NN', '.'], 'stanford_ner': ['PERSON', 'PERSON', 'O', 'O', 'DATE', 'DATE', 'DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'O', 'O', 'O', 'O', 'O', 'O', 'NUMBER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], 'stanford_head': [2, 3, 0, 5, 3, 7, 3, 9, 3, 13, 13, 13, 9, 15, 13, 15, 3, 3, 20, 18, 23, 23, 18, 25, 23, 3, 3, 32, 32, 32, 32, 27, 34, 27, 34, 34, 34, 40, 40, 37, 3], 'stanford_deprel': ['compound', 'nsubj', 'ROOT', 'case', 'nmod', 'amod', 'nmod:tmod', 'mark', 'xcomp', 'det', 'compound', 'compound', 'dobj', 'punct', 'appos', 'punct', 'punct', 'xcomp', 'det', 'dobj', 'case', 'nummod', 'nmod', 'case', 'nmod', 'punct', 'xcomp', 'amod', 'compound', 'compound', 'compound', 'dobj', 'mark', 'xcomp', 'dobj', 'cc', 'conj', 'det', 'compound', 'dobj', 'punct']}
        data2 = {'id': '61b3a65fb9b7111c4ca4', 'docid': 'NYT_ENG_20071026.0056.LDC2009T13', 'relation': 'no_relation', 'token': ['In', '1983', ',', 'a', 'year', 'after', 'the', 'rally', ',', 'Forsberg', 'received', 'the', 'so-called', '``', 'genius', 'award', "''", 'from', 'the', 'John', 'D.', 'and', 'Catherine', 'T.', 'MacArthur', 'Foundation', '.'], 'subj_start': 9, 'subj_end': 9, 'obj_start': 19, 'obj_end': 20, 'subj_type': 'PERSON', 'obj_type': 'PERSON', 'stanford_pos': ['IN', 'CD', ',', 'DT', 'NN', 'IN', 'DT', 'NN', ',', 'NNP', 'VBD', 'DT', 'JJ', '``', 'NN', 'NN', "''", 'IN', 'DT', 'NNP', 'NNP', 'CC', 'NNP', 'NNP', 'NNP', 'NNP', '.'], 'stanford_ner': ['O', 'DATE', 'O', 'DURATION', 'DURATION', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'O'], 'stanford_head': [2, 11, 11, 5, 11, 8, 8, 5, 11, 11, 0, 16, 16, 16, 16, 11, 16, 21, 21, 21, 16, 21, 26, 26, 26, 21, 11], 'stanford_deprel': ['case', 'nmod', 'punct', 'det', 'nmod:tmod', 'case', 'det', 'nmod', 'punct', 'nsubj', 'ROOT', 'det', 'amod', 'punct', 'compound', 'dobj', 'punct', 'case', 'det', 'compound', 'nmod', 'cc', 'compound', 'compound', 'compound', 'conj', 'punct']}
        data3 = {'id': '61b3a65fb9aa5e1bf6b0', 'docid': 'LTW_ENG_20070530.0085.LDC2009T13', 'relation': 'no_relation', 'token': ['He', 'received', 'an', 'undergraduate', 'degree', 'from', 'Morgan', 'State', 'University', 'in', '1950', 'and', 'applied', 'for', 'admission', 'to', 'graduate', 'school', 'at', 'the', 'University', 'of', 'Maryland', 'in', 'College', 'Park', '.'], 'subj_start': 0, 'subj_end': 0, 'obj_start': 20, 'obj_end': 25, 'subj_type': 'PERSON', 'obj_type': 'ORGANIZATION', 'stanford_pos': ['PRP', 'VBD', 'DT', 'JJ', 'NN', 'IN', 'NNP', 'NNP', 'NNP', 'IN', 'CD', 'CC', 'VBD', 'IN', 'NN', 'TO', 'VB', 'NN', 'IN', 'DT', 'NNP', 'IN', 'NNP', 'IN', 'NNP', 'NNP', '.'], 'stanford_ner': ['O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'O', 'DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'ORGANIZATION', 'O'], 'stanford_head': [2, 0, 5, 5, 2, 9, 9, 9, 5, 11, 2, 2, 2, 15, 13, 17, 15, 17, 21, 21, 17, 23, 21, 26, 26, 21, 2], 'stanford_deprel': ['nsubj', 'ROOT', 'det', 'amod', 'dobj', 'case', 'compound', 'compound', 'nmod', 'case', 'nmod', 'cc', 'conj', 'case', 'nmod', 'mark', 'acl', 'dobj', 'case', 'det', 'nmod', 'case', 'nmod', 'case', 'compound', 'nmod', 'punct']}


        data1_processed = convert_tacred_dict(data1)
        data2_processed = convert_tacred_dict(data2)
        data3_processed = convert_tacred_dict(data3)


        self.assertEqual(' '.join(data1_processed['e1']), 'All Basotho Convention')
        self.assertEqual(' '.join(data1_processed['e2']), 'Tom Thabane')

        self.assertEqual(' '.join(data2_processed['e1']), 'Forsberg')
        self.assertEqual(' '.join(data2_processed['e2']), 'John D.')

        self.assertEqual(' '.join(data3_processed['e1']), 'He')
        self.assertEqual(' '.join(data3_processed['e2']), 'University of Maryland in College Park')

    def test_convert_fewrel_dataset(self):
        data1 = {'relation': 'P931', 'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '(', 'TJQ', ')', '.'], 'h': ['tjq', 'Q1331049', [[16]]], 't': ['tanjung pandan', 'Q3056359', [[13, 14]]]}
        data2 = {'relation': 'P931', 'tokens': ['The', 'name', 'was', 'at', 'one', 'point', 'changed', 'to', 'Nottingham', 'East', 'Midlands', 'Airport', 'so', 'as', 'to', 'include', 'the', 'name', 'of', 'the', 'city', 'that', 'is', 'supposedly', 'most', 'internationally', 'recognisable', ',', 'mainly', 'due', 'to', 'the', 'Robin', 'Hood', 'legend', '.'], 'h': ['east midlands airport', 'Q8977', [[9, 10, 11]]], 't': ['nottingham', 'Q41262', [[8]]]}

        data1_processed = convert_fewrel_dict(data1)
        data2_processed = convert_fewrel_dict(data2)

        self.assertEqual(' '.join(data1_processed['e1']), 'TJQ')
        self.assertEqual(' '.join(data1_processed['e2']), 'Tanjung Pandan')

        self.assertEqual(' '.join(data2_processed['e1']), 'East Midlands Airport')
        self.assertEqual(' '.join(data2_processed['e2']), 'Nottingham')



if __name__ == '__main__':
    unittest.main()
