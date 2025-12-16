"""
AIbrain_TymE - Vylepšená neuronová síť pro řízení autíčka
Tým E - 4IT534

Architektura: 3-vrstvý MLP s podporou rychlosti jako vstupu
- 10 vstupů (9 raycast + rychlost) → 12 → 8 → 4

Vylepšení v2:
- Rychlost jako vstup (auto ví jak rychle jede)
- Lepší fitness funkce (bonus za rychlost)
- Adaptivní mutace (snižuje se s generacemi)
"""

from numpy import random as np_random
import random
import numpy as np
import copy
import string

# === KONFIGURACE ARCHITEKTURY ===

N_INPUTS = 10      # 9 raycast paprsků + 1 rychlost
N_HIDDEN1 = 12     # První skrytá vrstva (větší pro více vstupů)
N_HIDDEN2 = 8      # Druhá skrytá vrstva
N_ACTIONS = 4      # [up, down, left, right]

# Normalizace vstupů
MAX_RAY_DISTANCE = 3.0  # Raycast vrací vzdálenost v dlaždicích
MAX_SPEED = 500.0        # Maximální rychlost auta (z constants.py)

# Mutace - adaptivní
MUTATION_RATE_INITIAL = 0.1   # Počáteční mutation rate
MUTATION_RATE_MIN = 0.01       # Minimální mutation rate
MUTATION_DECAY = 0.997         # Jak rychle se snižuje (každou mutací)


class AIbrain_TymE:
    def __init__(self):
        super().__init__()
        self.score = 0
        self.chars = string.ascii_letters + string.digits
        self.decider = 0
        
        # Data z auta (aktualizuje se přes passcardata)
        self.x = 0
        self.y = 0
        self.speed = 0
        
        # Adaptivní mutace
        self.mutation_rate = MUTATION_RATE_INITIAL
        self.generation = 0
        
        self.init_param()

    def init_param(self):
        """
        Inicializace 3-vrstvé sítě s 10 vstupy.
        Použije Xavier/He inicializaci pro lepší konvergenci.
        """
        
        # Vrstva 1: vstup (10) -> skrytá1 (12)
        # He inicializace: * sqrt(2/n_inputs)
        self.W1 = np_random.randn(N_HIDDEN1, N_INPUTS) * np.sqrt(2.0 / N_INPUTS)
        self.b1 = np.zeros(N_HIDDEN1)
        
        # Vrstva 2: skrytá1 (12) -> skrytá2 (8)
        self.W2 = np_random.randn(N_HIDDEN2, N_HIDDEN1) * np.sqrt(2.0 / N_HIDDEN1)
        self.b2 = np.zeros(N_HIDDEN2)
        
        # Vrstva 3: skrytá2 (8) -> výstup (4)
        self.W3 = np_random.randn(N_ACTIONS, N_HIDDEN2) * np.sqrt(2.0 / N_HIDDEN2)
        self.b3 = np.zeros(N_ACTIONS)
        
        # Bias pro plyn - mírně kladný (auto má tendenci jet)
        self.b3[0] = 0.5
        
        self.NAME = "TymE_v2_" + ''.join(random.choices(self.chars, k=4))
        self.mutation_rate = MUTATION_RATE_INITIAL
        self.generation = 0
        self.store()

    def _relu(self, x):
        """ReLU aktivace - rychlá a efektivní"""
        return np.maximum(0, x)
    
    def _leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU - zabraňuje 'mrtvým' neuronům"""
        return np.where(x > 0, x, alpha * x)

    def _sigmoid(self, x):
        """Sigmoid - výstup vždy mezi 0 a 1"""
        x = np.clip(x, -10, 10)  # prevence overflow
        return 1.0 / (1.0 + np.exp(-x))

    def decide(self, data):
        """
        Forward pass - rozhodnutí na základě raycastů a rychlosti.
        """
        self.decider += 1
        
        # Vstup: raycast vzdálenosti (v dlaždicích)
        rays = np.asarray(data, dtype=float).ravel()
        
        # Normalizace raycastů do [0, 1]
        rays_norm = np.clip(rays / MAX_RAY_DISTANCE, 0, 1)
        
        # Zajistit 9 raycast hodnot
        if rays_norm.size < 9:
            rays_norm = np.concatenate([rays_norm, np.zeros(9 - rays_norm.size)])
        elif rays_norm.size > 9:
            rays_norm = rays_norm[:9]
        
        # Přidat normalizovanou rychlost jako 10. vstup
        speed_norm = np.clip(self.speed / MAX_SPEED, 0, 1)
        x = np.concatenate([rays_norm, [speed_norm]])
        
        # 3-vrstvý forward pass s Leaky ReLU
        z1 = self.W1.dot(x) + self.b1
        a1 = self._leaky_relu(z1)
        
        z2 = self.W2.dot(a1) + self.b2
        a2 = self._leaky_relu(z2)
        
        z3 = self.W3.dot(a2) + self.b3
        output = self._sigmoid(z3)
        
        return output

    def mutate(self):
        """
        Adaptivní mutace - mutation rate se postupně snižuje.
        Obsahuje jak malé perturbace, tak občasné větší skoky.
        """
        self.generation += 1
        
        # Adaptivní snižování mutation rate
        self.mutation_rate = max(
            MUTATION_RATE_MIN,
            self.mutation_rate * MUTATION_DECAY
        )
        
        mr = self.mutation_rate
        
        # Mutace všech vrstev
        self.W1 += np_random.randn(*self.W1.shape) * mr
        self.b1 += np_random.randn(*self.b1.shape) * mr
        self.W2 += np_random.randn(*self.W2.shape) * mr
        self.b2 += np_random.randn(*self.b2.shape) * mr
        self.W3 += np_random.randn(*self.W3.shape) * mr
        self.b3 += np_random.randn(*self.b3.shape) * mr
        
        # Občasná velká mutace (explorace, 5% šance)
        if np_random.rand() < 0.05:
            layer = np_random.choice(['W1', 'W2', 'W3'])
            W = getattr(self, layer)
            i = np_random.randint(0, W.shape[0])
            j = np_random.randint(0, W.shape[1])
            W[i, j] += np_random.randn() * 0.25  # Větší skok
        
        # Občasný reset jednoho neuronu (2% šance) - pomáhá uniknout lokálním minimům
        if np_random.rand() < 0.02:
            layer_idx = np_random.choice([1, 2, 3])
            if layer_idx == 1:
                neuron = np_random.randint(0, self.W1.shape[0])
                self.W1[neuron, :] = np_random.randn(self.W1.shape[1]) * 0.3
                self.b1[neuron] = 0
            elif layer_idx == 2:
                neuron = np_random.randint(0, self.W2.shape[0])
                self.W2[neuron, :] = np_random.randn(self.W2.shape[1]) * 0.3
                self.b2[neuron] = 0

        self.NAME += "m"
        self.store()

    def store(self):
        """Uloží všechny parametry do slovníku."""
        self.parameters = copy.deepcopy({
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'NAME': self.NAME,
            'mutation_rate': self.mutation_rate,
            'generation': self.generation,
        })

    def get_parameters(self):
        return copy.deepcopy(self.parameters)

    def set_parameters(self, parameters):
        """
        Načte parametry modelu.
        """
        if isinstance(parameters, np.lib.npyio.NpzFile):
            p = {key: parameters[key] for key in parameters.files}
        else:
            p = copy.deepcopy(parameters)
        
        # Kontrola, že jde o správný formát (3-vrstvý model s W3)
        if 'W3' not in p:
            print("VAROVÁNÍ: Nekompatibilní formát modelu (chybí W3).")
            print("Používám novou náhodnou inicializaci.")
            self.init_param()
            return
        
        self.W1 = np.array(p['W1'], dtype=float)
        self.b1 = np.array(p['b1'], dtype=float)
        self.W2 = np.array(p['W2'], dtype=float)
        self.b2 = np.array(p['b2'], dtype=float)
        self.W3 = np.array(p['W3'], dtype=float)
        self.b3 = np.array(p['b3'], dtype=float)
        
        self.NAME = str(p.get('NAME', 'TymE_loaded'))
        self.mutation_rate = float(p.get('mutation_rate', MUTATION_RATE_INITIAL))
        self.generation = int(p.get('generation', 0))
        
        self.parameters = p
        
        W1_shape = self.W1.shape
        W2_shape = self.W2.shape
        W3_shape = self.W3.shape
        print(f"Model načten: {self.NAME} ({W1_shape[1]}→{W1_shape[0]}→{W2_shape[0]}→{W3_shape[0]})")
        print(f"  Generace: {self.generation}, mutation_rate: {self.mutation_rate:.4f}")

    def calculate_score(self, distance, time, no):
        """
        Vylepšená fitness funkce.
        
        Komponenty:
        - Základní skóre = vzdálenost (hlavní cíl)
        - Bonus za rychlost = odměna za průměrnou rychlost
        - Mírná penalizace za čas = motivace být rychlejší
        """
        # Základní skóre = vzdálenost (v dlaždicích)
        base_score = distance
        
        # Bonus za průměrnou rychlost
        if time > 0.5:  # Minimální čas pro výpočet
            avg_speed = distance / time
            # Normalizovaný bonus (max ~2 body za maximální rychlost)
            speed_bonus = avg_speed * 0.3
        else:
            speed_bonus = 0
        
        # Mírná penalizace za čas (motivace dokončit rychleji)
        # Ale ne příliš velká, aby auto neriskovalo příliš
        time_penalty = time * 0.02
        
        # Celkové skóre
        self.score = base_score + speed_bonus - time_penalty
        
        # Uložit pořadí pro případné další využití
        self.no = no

    def passcardata(self, x, y, speed):
        """Přijímá data o pozici a rychlosti auta."""
        self.x = x
        self.y = y
        self.speed = speed

    def getscore(self):
        return self.score

    def get_model_info(self):
        """Vrátí informace o modelu pro debugging."""
        arch = f"{N_INPUTS}→{N_HIDDEN1}→{N_HIDDEN2}→{N_ACTIONS}"
        params = (self.W1.size + self.b1.size + 
                 self.W2.size + self.b2.size + 
                 self.W3.size + self.b3.size)
        
        return {
            'name': self.NAME,
            'architecture': arch,
            'total_parameters': params,
            'mutation_rate': self.mutation_rate,
            'generation': self.generation,
        }
