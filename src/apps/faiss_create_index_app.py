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


# python -m src.apps.faiss_create_index_app --path config/faiss_index_creation.yaml
if __name__ == "__main__":
    from gensim.models import KeyedVectors

    config = Config.parse_args_and_get_config().get('faiss_create_index_app')
    gensim_model = KeyedVectors.load_word2vec_format(**config.get('gensim_model'))
    create_index(gensim_model, config.get('index_save_path'), config.get('vocab_save_path'))



