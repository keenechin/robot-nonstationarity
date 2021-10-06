import numpy as np


class DummyCamera():
    def __init__(self):
        pass

    def get_feed(self):
        return self

    def get(self):
        return [np.random.randint(1, 400), np.random.randint(1, 600),
                np.random.randint(1, 400), np.random.randint(1, 600)]
