from src.dataprocessing.semeval.dataset_converter import convert_semeval_dict
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



if __name__ == '__main__':
    unittest.main()
