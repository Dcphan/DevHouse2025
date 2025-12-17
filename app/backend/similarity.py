from numpy.linalg import norm
import numpy as np

class Similarity():
    def cosine_sim(self, a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    def match_embedding(self, emb):
        best_id = None
        best_score = -1

        for person_id, db_emb in self.EMBEDDING_DB.items():
            score = self.cosine_sim(emb, db_emb)
            if score > best_score:
                best_score = score
                best_id = person_id

        return best_id, float(best_score)
