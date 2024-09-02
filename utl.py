# Slot car utilities

import pickle


def p_load(name: str):
    with open(f'pickles/{name}.pkl', 'rb') as f:
        return pickle.load(f)


def p_save(element, name: str):
    with open(f'pickles/{name}.pkl', 'wb') as f:
        return pickle.dump(element, f)
