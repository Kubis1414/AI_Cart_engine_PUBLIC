"""
AIbrain_TymE - Neuronová síť pro řízení autíčka
Tým E - 4IT534

Architektura: 3-vrstvý MLP (odpovídá uloženému souboru)
- Vstup: 9 raycast paprsků
- Skrytá vrstva 1: 16 neuronů
- Skrytá vrstva 2: 8 neuronů
- Výstup: 4 akce (plyn, brzda, doleva, doprava)
"""

from numpy import random as np_random
import random
import numpy as np
import copy
import string

# Architektura sítě - MUSÍ odpovídat uloženému souboru!
N_INPUTS = 9       # 9 raycast paprsků
N_HIDDEN1 = 16     # První skrytá vrstva
N_HIDDEN2 = 8      # Druhá skrytá vrstva
N_ACTIONS = 4      # [up, down, left, right]

# DŮLEŽITÉ: Raycast vrací vzdálenost v DLAŽDICÍCH, ne pixelech!
MAX_RAY_DISTANCE = 15.0

# Mutace
MUTATION_RATE = 0.15


class v0_1:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0
        
        # Data z auta
        self.x = 0
        self.y = 0
        self.speed = 0
        
        self.init_param()

    def init_param(self):
        """
        Inicializace 3-vrstvé sítě.
        """
        # Vrstva 1: vstup (9) -> skrytá1 (16)
        self.W1 = np_random.randn(N_HIDDEN1, N_INPUTS) * 0.3
        self.b1 = np.zeros(N_HIDDEN1)
        
        # Vrstva 2: skrytá1 (16) -> skrytá2 (8)
        self.W2 = np_random.randn(N_HIDDEN2, N_HIDDEN1) * 0.3
        self.b2 = np.zeros(N_HIDDEN2)
        
        # Vrstva 3: skrytá2 (8) -> výstup (4)
        self.W3 = np_random.randn(N_ACTIONS, N_HIDDEN2) * 0.3
        # Bias: [plyn, brzda, left, right]
        self.b3 = np.zeros(N_ACTIONS)
        
        self.NAME = "TymE_" + ''.join(random.choices(self.chars, k=4))
        self.store()

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        """Sigmoid - výstup vždy mezi 0 a 1"""
        x = np.clip(x, -10, 10)  # prevence overflow
        return 1.0 / (1.0 + np.exp(-x))

    def decide(self, data):
        """
        Forward pass - 3 vrstvy s ReLU a sigmoid na výstupu.
        """
        self.decider += 1
        
        # Vstup: raycast vzdálenosti (v dlaždicích)
        x = np.asarray(data, dtype=float).ravel()
        
        # Zajistit správnou délku
        if x.size < N_INPUTS:
            x = np.concatenate([x, np.zeros(N_INPUTS - x.size)])
        elif x.size > N_INPUTS:
            x = x[:N_INPUTS]
        
        # Normalizace do [0, 1]
        x_norm = x / MAX_RAY_DISTANCE
        x_norm = np.clip(x_norm, 0, 1)
        
        # Forward pass
        z1 = self.W1.dot(x_norm) + self.b1
        a1 = self._relu(z1)
        
        z2 = self.W2.dot(a1) + self.b2
        a2 = self._relu(z2)
        
        z3 = self.W3.dot(a2) + self.b3
        output = self._sigmoid(z3)
        
        return output

    def mutate(self):
        """
        Mutace - přidej malý šum ke všem vahám.
        """
        self.W1 += np_random.randn(*self.W1.shape) * MUTATION_RATE
        self.b1 += np_random.randn(*self.b1.shape) * MUTATION_RATE
        self.W2 += np_random.randn(*self.W2.shape) * MUTATION_RATE
        self.b2 += np_random.randn(*self.b2.shape) * MUTATION_RATE
        self.W3 += np_random.randn(*self.W3.shape) * MUTATION_RATE
        self.b3 += np_random.randn(*self.b3.shape) * MUTATION_RATE
        
        # Občasná velká mutace (5% šance)
        if np_random.rand() < 0.05:
            layer = np_random.choice(['W1', 'W2', 'W3'])
            W = getattr(self, layer)
            i, j = np_random.randint(0, W.shape[0]), np_random.randint(0, W.shape[1])
            W[i, j] += np_random.randn() * 0.5

        self.NAME += "m"
        self.store()

    def store(self):
        self.parameters = copy.deepcopy({
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'NAME': self.NAME,
        })

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            p = {key: parameters[key] for key in parameters.files}
        else:
            p = copy.deepcopy(parameters)
        
        # Kontrola kompatibility
        try:
            W1_shape = np.array(p['W1']).shape
            W2_shape = np.array(p['W2']).shape
            W3_shape = np.array(p['W3']).shape
            
            expected = {
                'W1': (N_HIDDEN1, N_INPUTS),
                'W2': (N_HIDDEN2, N_HIDDEN1),
                'W3': (N_ACTIONS, N_HIDDEN2),
            }
            
            if W1_shape != expected['W1'] or W2_shape != expected['W2'] or W3_shape != expected['W3']:
                print(f"VAROVÁNÍ: Nekompatibilní váhy v uloženém souboru!")
                print(f"  Očekáváno: W1={expected['W1']}, W2={expected['W2']}, W3={expected['W3']}")
                print(f"  Nalezeno: W1={W1_shape}, W2={W2_shape}, W3={W3_shape}")
                print(f"  Používám novou náhodnou inicializaci.")
                self.init_param()
                return
        except KeyError as e:
            print(f"VAROVÁNÍ: Chybí klíč {e} v uloženém souboru, používám náhodnou inicializaci.")
            self.init_param()
            return
        
        self.parameters = p
        self.W1 = np.array(p['W1'], dtype=float)
        self.b1 = np.array(p['b1'], dtype=float)
        self.W2 = np.array(p['W2'], dtype=float)
        self.b2 = np.array(p['b2'], dtype=float)
        self.W3 = np.array(p['W3'], dtype=float)
        self.b3 = np.array(p['b3'], dtype=float)
        self.NAME = str(p['NAME'])

    def calculate_score(self, distance, time, no):
        """Skóre = ujetá vzdálenost. Jednoduché a funkční."""
        self.score = distance

    def passcardata(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score
