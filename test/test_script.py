import timeit
import unittest
import fasttext_parallel as ft
import fasttext as ft_ref
import logging
import csv

logging.basicConfig(level=logging.ERROR)
ft_ref.FastText.eprint = lambda x: None
MODEL_PATH = "./model/lid.176.bin"


def text_iter():
    texts = []
    with open("./data/train.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            texts.append(line[1])
    return texts


class TestFastText(unittest.TestCase):
    model = ft.load_model(MODEL_PATH)
    model_ref = ft_ref.load_model(MODEL_PATH)

    def test_get_labels(self):
        self.assertSetEqual(
            set(self.model.get_labels().values()),
            set(self.model_ref.get_labels())
        )

    def test_simple(self):
        k = 2
        test_text = ["你好", "春天在哪里", "吃了吗", "hello", "how are you"]
        labels, probs = self.model.batch(test_text, k, -1.0)
        labels_ref, probs_ref = self.model_ref.predict(test_text, k=k)
        for j in range(k):
            for i in range(len(labels)):
                self.assertEqual(self.model.get_label_by_id(labels[i][j]), labels_ref[i][j])
                self.assertAlmostEqual(probs[i][j], probs_ref[i][j], 1)

    def test_benchmark(self):
        k = 2
        texts = text_iter()
        time = timeit.timeit(lambda: self.model.batch(texts, k, -1.0), number=10)
        time_ref = timeit.timeit(lambda: self.model_ref.predict(texts, k, -1.0), number=10)
        print(f"time taken {time}")
        print(f"ref time taken {time_ref}")


if __name__ == '__main__':
    unittest.main()
