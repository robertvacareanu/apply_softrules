from src.config import Config
import faiss
import numpy as np
import pickle

def create_index(gensim_model, index_save_path, vocab_save_path):
    vectors = gensim_model.vectors / np.linalg.norm(gensim_model.vectors)
    index   = faiss.IndexFlatL2(vectors.shape[1])
    vocab   = gensim_model.index_to_key
    index.add(vectors)
    with open(vocab_save_path, 'wb+') as fout:
        pickle.dump(vocab, fout)
    faiss.write_index(index, index_save_path)

"""
Create a faiss index for the word embeddings model
Needs to know the necessary parameters for gensim:
- fname
- binary
- no_header
Additionally, two paths must be given:
- where to save the index
- where to save the vocabulary
"""
# python -m src.apps.preprocessing.faiss_create_index_app --path config/base_config.yaml config/faiss_index_creation.yaml
if __name__ == "__main__":
    from gensim.models import KeyedVectors

    config = Config.parse_args_and_get_config()#.get('faiss_create_index_app')
    glove_dict = {
        'fname'    : config.get('gensim_model').get_path('fname'),
        'binary'   : config.get('gensim_model').get('binary'),
        'no_header': config.get('gensim_model').get('no_header'),
    }
    gensim_model = KeyedVectors.load_word2vec_format(**glove_dict)
    create_index(gensim_model, config.get_path('index_save_path'), config.get_path('vocab_save_path'))



