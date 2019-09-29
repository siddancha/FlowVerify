import numpy as np


class Scorer:
    def __init__(self, list_score_names):
        self.score_name = 'score'
        self.list_score_names = list_score_names
        self.params = None

    def num_scores(self):
        return len(self.list_score_names)

    def get_full_score_name(self):
        return '_'.join([self.score_name] + self.list_score_names)

    def score(self, list_score):
        pass


class HardAND(Scorer):
    def __init__(self, list_score_names):
        super(HardAND, self).__init__(list_score_names)
        self.score_name = 'HardAND'

    def score(self, list_score, params):
        return float(np.prod(list_score > params))


class SoftAND(Scorer):
    def __init__(self, list_score_names):
        super(SoftAND, self).__init__(list_score_names)
        self.score_name = 'SoftAND'

    def score(self, list_score, params):
        exponent = np.exp(params)
        return np.prod(np.array(list_score) ** exponent)
