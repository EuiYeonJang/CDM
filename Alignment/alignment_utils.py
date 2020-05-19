import numpy as np
import statsmodels.api as sm
import pickle as pkl
from collections import Counter

def prime_lists(target_gender):
    """
    Parameters
        target_gender (str): "m" or "f"

    Returns
        tuple of original splits (split with female prime, split with male prime)
    """
    
    with open("../Corpora/ICSI/in_between_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    if target_gender == "m":
        return fm, mm
    else:
        return ff, mf


def create_vocab(adj_pair_list):
    """
    Parameters: 
        adj_pair_list -list of tuples (gender of prime, prime counter, target counter)
    
    Returns:
        set of vocab
    """
    vocab = set()

    for _, prime, target in adj_pair_list:
        vocab |= set(prime.keys())
        vocab |= set(target.keys())

    return vocab

class LexicalAlignmentOne():
    def __init__(self, target_name, female_prime_list, male_prime_list):
        self.target_name = target_name
        print("prepping")
        self.__prep_list(female_prime_list, male_prime_list)
        print("creating vocab")
        self.__create_vocab()
        print("creating features and target")
        self.__create_X_y()


    def __prep_list(self, female_prime_list, male_prime_list):
        self.adj_pair_list = list()

        for pair in female_prime_list:
            combined = pair["a"] + pair["fb"]
            if sum(combined.values()) > 0 and sum(pair["b"].values()) > 0:
                self.adj_pair_list.append(("f", combined , pair["b"]))


        for pair in male_prime_list:
            combined = pair["a"] + pair["mb"]
            if sum(combined.values()) > 0 and sum(pair["b"].values()) > 0:
                self.adj_pair_list.append(("m",  combined, pair["b"]))


    def __create_vocab(self):
        self.vocab = set()

        for _, prime, target in self.adj_pair_list:
            self.vocab |= set(prime.keys())
            self.vocab |= set(target.keys())

        print(len(self.vocab))
        # with open(f"{self.target_name}_alignment_one_vocab.pkl", "wb") as f:
        #     pkl.dump(self.vocab, f)

    
    def __create_X_y(self):
        self.c_count = {w: list() for w in self.vocab}
        self.c_gender = {w: list() for w in self.vocab}
        self.y = {w: list() for w in self.vocab}

        for w in self.vocab:
            self.c_count[w] = [prime[w] / sum(prime.values()) for _, prime, _ in self.adj_pair_list]
            self.c_gender[w] = [ 1 if prime_gender == "m" else 0 for prime_gender, _, _ in self.adj_pair_list]
            self.y[w] = [ 1 if target[w] > 0 else 0 for _, _, target in self.adj_pair_list]

        


    def calculate_beta(self):
        pvalue_dict = {w: {i: "undefined" for i in range(4)} for w in self.vocab}
        zscores_dict = {w: {i: "undefined" for i in range(4)} for w in self.vocab}

        for w in self.vocab:
            c_w = self.c_count[w]
            g_w = self.c_power[w]
            y_w = self.y[w]

            if sum(y_w) > 0:
                c_W = np.array(c_w)
                g_w = np.array(g_w)
                X = np.array([np.ones(len(c_W)), c_w, g_w, c_w*g_w]).T

                y_w = np.array(y_w)

                res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
                
                p_values = res.pvalues
                for i, p in enumerate(p_values):
                    pvalue_dict[w][i] = p

                z_scores = res.tvalues
                for i, z in enumerate(z_scores):
                    zscores_dict[w][i] = z
                    

        with open(f"{self.target_name}_pvalues.pkl", "wb") as f:
            pkl.dump(pvalue_dict, f)

        with open(f"{self.target_name}_zscores.pkl", "wb") as f:
            pkl.dump(zscores_dict, f)
                
