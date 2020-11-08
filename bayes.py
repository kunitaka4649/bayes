class Bayes():
    def  __init__(self):
        self.nums = 0 # 総単語数（重複あり）
        self.unions = {} # ある集合の総単語数（一次元dict）
        self.words = {} # ある集合のある単語の数（二次元dict)
        self.log_probs = None

    def _prob_if_unknown_word(self, token, union):
        # p(unknown_word | union)を計算する
        return 0.1 / self.unions[union]

    def train(self, tokens, union):
        for token in tokens:
            # 総単語数をふやす
            self.nums += 1
            # ある集合の数をふやす
            self.unions[union] = self.unions.get(union, 0) + 1
            # ある集合の数の単語の数をふやす
            self.words[union] = self.words.get(union, {})
            self.words[union][token] = self.words[union].get(token, 0) + 1

    def predict(self, tokens):
        probs = {}
        for union in self.unions:
            prob = self.unions[union] / self.nums
            for token in tokens:
                if token not in self.words[union]:
                    prob *= self._prob_if_unknown_word(token, union)
                    continue
                prob *= self.words[union][token] / self.unions[union]
            probs[union] = prob
        self.log_probs = probs
        return max(probs, key=probs.get)
    
    def debug(self):
        print(self.nums)
        print(self.unions)
        print(self.words)