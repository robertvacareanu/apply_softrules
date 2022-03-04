import unittest
from src.rulegeneration.simple_rule_generation import word_rule

class TestRuleGeneration(unittest.TestCase):

    def test_simple_rule_generation(self):
        data1 = {'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '(', 'TJQ', ')', '.'], 'e1_start': 16, 'e1_end': 17, 'e2_start': 13, 'e2_end': 15, 'e1': ['TJQ'], 'e2': ['Tanjung', 'Pandan'], 'relation': 'P931'}
        data2 = {'tokens': ['The', 'system', 'as', 'described', 'above', 'has', 'its', 'greatest', 'application', 'in', 'an', 'arrayed', 'configuration', 'of', 'antenna', 'elements', '.'], 'e1_start': 12, 'e1_end': 13, 'e2_start': 15, 'e2_end': 16, 'e1': ['configuration'], 'e2': ['elements'], 'relation': 3}
        data3 = {'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', '"', 'TJQ', ')', '.'], 'e1_start': 16, 'e1_end': 17, 'e2_start': 13, 'e2_end': 15, 'e1': ['TJQ'], 'e2': ['Tanjung', 'Pandan'], 'relation': 'P931'}
        data4 = {'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight', 'to', 'Tanjung', 'Pandan', "'", 'TJQ', ')', '.'], 'e1_start': 16, 'e1_end': 17, 'e2_start': 13, 'e2_end': 15, 'e1': ['TJQ'], 'e2': ['Tanjung', 'Pandan'], 'relation': 'P931'}

        rule1 = word_rule(data1)
        rule2 = word_rule(data2)
        rule3 = word_rule(data3)
        rule4 = word_rule(data4)

        self.assertEqual(str(rule1), """[word="("]""")
        self.assertEqual(str(rule2), """[word=of] [word=antenna]""")
        self.assertEqual(str(rule3), """[word="\\""]""")
        self.assertEqual(str(rule4), """[word="'"]""")

if __name__ == '__main__':
    unittest.main()
