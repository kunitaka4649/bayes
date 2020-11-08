from bayes import Bayes
from configure import Configure

import pickle
from nltk.tokenize import word_tokenize


class InputManager():
    def __init__(self, configure):
        self.is_train  = configure.get_is_train()
        self.load_path = configure.get_load_path()
        self.save_path = configure.get_save_path()
        self.targets, self.sents = configure.input_parser(configure.get_input_path())

    def can_load_model(self):
        return len(self.load_path) != 0


def prep_model(can_load_model, load_path):
    bayes = None
    if can_load_model:
        with open(load_path, "rb") as f:
            bayes = pickle.load(f)
    else:
        bayes = Bayes()
    return bayes


def train_or_predict(bayes, targets, sents, is_train):
    for idx, target in enumerate(targets):
        sent = sents[idx]
        tokens = word_tokenize(sent)
        if is_train:
            bayes.train(tokens, target)
        else:
            if idx:
                print()
            print("入力文書：", sent)
            print("推論結果：", bayes.predict(tokens))
            print("スコア", bayes.log_probs)
    return bayes


def main():
    configure = Configure("config.ini")
    i_m = InputManager(configure)

    # モデルの読み込み or 新規作成
    bayes = prep_model(i_m.can_load_model(), i_m.load_path)
    # 学習 or 推論
    bayes = train_or_predict(bayes, i_m.targets, i_m.sents, i_m.is_train)
    # モデルの保存
    with open(i_m.save_path, "wb") as f:
        pickle.dump(bayes, f)


if  __name__ == "__main__":
    main()