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



if __name__ == "__main__":
    from gensim.models import KeyedVectors
    gensim_model = KeyedVectors.load_word2vec_format(**{'fname': '/data/nlp/corpora/softrules/models/glove.6B.50d.txt',    'binary': False, 'no_header': True})
    create_index(gensim_model, "/data/nlp/corpora/softrules/faiss_index/glove.6B.50d_index", "/data/nlp/corpora/softrules/faiss_index/glove.6B.50d_vocab")



