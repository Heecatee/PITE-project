import gym
from gym import spaces, logger
from gym.utils import seeding

help(gym)

class SwarmBall(gym.Env):
    def __init__(self):
        """
            Tutaj Wiolu wpisz stałe fizyczne środowiska
            a także obiektów (typu masa bota) na zasadzie:
            self.gravity = 9.8
        """

    def seed(self, seed=None):
        """
            To przerażające nasiono!!!
        """
        pass

    def step(self, action):
        """
            Tutaj będzie jedna klatka animacji. Wszystkie obliczenia fizyczne będą się tu odbywać.
        """
        pass

    def reset(self):
        """
            Generuje stan początkowy.
        """
        pass

    def render(self):
        """
            Rysowanie klatki
        """
        pass

    def close(self):
        """
            Zamknięcie środowiska.
        """
        pass