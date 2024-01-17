import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm

'''
It can be seen as a directed graph where the market states are the vertices and 
the directed edges give the transition probability
'''
states = [0, 1, 2]        # ["bull", "bear", "stagnant"]

'''
The probabilities can be arranged into lists: each probability list has: 
    · the prob. of staying in the same state (index 0)
    · the prob. of going to the state to the right (index 1) - see `states` list
    · the prob. of going to the state to the left (index 2 - OMITTED)
'''
p_bull = [0.9, 0.075]   # no need for third prob ,0.025] thanks to normalization
p_bear = [0.8, 0.05]    # no need for third prob ,0.15] thanks to normalization
p_stag = [0.5, 0.25]    # no need for third prob ,0.25] thanks to normalization
edges = np.array([p_bull, p_bear, p_stag])

@njit
def evolve(init_state, n_iters):
    '''
    Given an initial market state (Bull: 0, Bear: 1, Stagnant: 2), and the
    maximum number of iterations, this method tracks the market state at
    each iteration. Returns a list of market states.
    '''
    h = [init_state]
    for _ in range(int(n_iters)):
        rv = random.random()
        today = h[-1]
        probs = edges[today]
        
        if rv < (1 - probs[0]):
            if rv < probs[1]:
                h.append((today + 1)%3) # look at the right
            else:
                h.append((today + 2)%3) # look at the left
        else:
            h.append(today)
    return h

def count_states(history):
    return dict((s, history.count(s)) for s in states)

def market_analysis(show_plot=False):
    df = pd.DataFrame(columns=["Iter", "Bull", "Bear", "Stagnant"])

    random.seed(42)

    iterations = np.logspace(1, 6, 100, dtype=int)
    for iter in tqdm(iterations):
        h = evolve(random.choice([0,1,2]), iter) # not efficient
        res = count_states(h)
        df.loc[len(df.index)] = [iter]+list(res.values()) # put the values into the dataframe saving also the number of iterations

    df.Bull /= df.Iter
    df.Bear /= df.Iter
    df.Stagnant /= df.Iter

    if show_plot:
        plt.title("Market states convergence")

        plt.semilogx(df.Iter, df.Bull, label="Bull")
        plt.semilogx(df.Iter, df.Bear, label="Bear")
        plt.semilogx(df.Iter, df.Stagnant, label="Stagnant")

        plt.xlabel("Iterations")
        plt.ylabel("State population density")

        plt.legend(loc='best', frameon=False)
        plt.show()
    else:
        return df

if __name__ == "__main__":
    market_analysis()