import pickle as pkl
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
  
# create vocab for subset
def create_vocab(subset, subset_name):
    vocab = set()
    for p, t in subset:
        vocab |= set(p.keys())
        vocab |= set(t.keys())

    with open(f"{subset_name}_vocab.pkl", "wb") as f:
        pkl.dump(vocab, f)

    return vocab


def create_bi_vocab(subset, subset_name):
    vocab = set()
    for _, p, t in subset:
        vocab |= set(p.keys())
        vocab |= set(t.keys())

    with open(f"{subset_name}_vocab.pkl", "wb") as f:
        pkl.dump(vocab, f)

    return vocab

# create the features and labels
def create_X_y(subset, subset_name, vocab):
    c_count = {w: list() for w in vocab}
    y = {w: list() for w in vocab}
    # p_len = {w: list() for w in vocab}

    for w in vocab:
        for p, t in subset:
            # c_count[w].append(p[w])
            c_count[w].append(p[w]/sum(p.values()))
            y[w].append(1 if t[w] > 0 else 0)
            # p_len[w].append(sum(p.values()))

    if subset_name == "mf" or subset_name == "mm":
        c_power = {w: [1]*len(c_count[w]) for w in c_count}
    else:
        c_power = {w: [0]*len(c_count[w]) for w in c_count}


    # d = dict(c_count=c_count, y=y, p_len=p_len, c_power=c_power)
    d = dict(c_count=c_count, y=y, c_power=c_power)
    with open(f"{subset_name}_X_y.pkl", "wb") as f:
        pkl.dump(d, f)

    return d


# create the features and labels
def create_bi_X_y(subset, subset_name, vocab):
    c_count = {w: list() for w in vocab}
    y = {w: list() for w in vocab}
    # p_len = {w: list() for w in vocab}
    c_power = {w: list() for w in vocab}

    for w in vocab:
        for pg, p, t in subset:
            # c_count[w].append(p[w])
            c_count[w].append(p[w]/sum(p.values()))
            y[w].append(1 if t[w] > 0 else 0)
            # p_len[w].append(sum(p.values()))
            c_power[w].append(1 if pg =="m" else 0)
            

    # d = dict(c_count=c_count, y=y, p_len=p_len, c_power=c_power)
    d = dict(c_count=c_count, y=y, c_power=c_power)

    with open(f"{subset_name}_X_y.pkl", "wb") as f:
        pkl.dump(d, f)

    return d


# calculate beta
def calculate_beta(X_dict, vocab, subset_name):
    c_count = X_dict["c_count"]
    y = X_dict["y"]
    # p_len = X_dict["p_len"]
    c_power = X_dict["c_power"]

    # beta_dict = {w: {i: "undefined" for i in range(7)} for w in vocab}
    pvalue_dict = {w: {i: "undefined" for i in range(4)} for w in vocab}
    zscores_dict = {w: {i: "undefined" for i in range(4)} for w in vocab}

    for w in vocab:
        cc_w = c_count[w]
        y_w = y[w]
        # p_len_w = p_len[w]
        cp_w = c_power[w]

        if sum(y_w) > 0:
            cc_w = np.array(cc_w)
            # p_len_w = np.array(p_len_w)
            cp_w = np.array(cp_w)
            # X = np.array([np.ones(len(cc_w)), cc_w, cp_w, p_len_w, cc_w*cp_w, cc_w*p_len_w, cp_w*p_len_w, cc_w*cp_w*p_len_w]).T
            X = np.array([np.ones(len(cc_w)), cc_w, cp_w, cc_w*cp_w]).T

            y_w = np.array(y_w)

            glm_binom = sm.GLM(y_w, X, family=sm.families.Binomial())
            res = glm_binom.fit()
            p_values = res.pvalues
            for i, p in enumerate(p_values):
                pvalue_dict[w][i] = p

            z_scores = res.tvalues
            for i, z in enumerate(z_scores):
                zscores_dict[w][i] = z
                

    with open(f"{subset_name}_pvalues.pkl", "wb") as f:
        pkl.dump(pvalue_dict, f)

    with open(f"{subset_name}_zscores.pkl", "wb") as f:
        pkl.dump(zscores_dict, f)
    


if __name__ == "__main__":
    N_GROUP = 2
    print("load subsets\n")
    
    if N_GROUP == 2:
        # with open("parsed_target_split.pkl", 'rb') as f:
        with open("parsed_prime_split.pkl", 'rb') as f:
            subsets = pkl.load(f)

        # names = ["m", "f"]
        names = ["m"]

        for subset_name in names:
            print(subset_name)
            subset = subsets[subset_name]

            print("\tcreating vocab")
            vocab = create_bi_vocab(subset, subset_name)
            # print("\tload vocab")
            # with open(f"{subset_name}_vocab.pkl", "rb") as f:
            #     vocab = pkl.load(f)

            print("\tcreating features and target")
            X_dict = create_bi_X_y(subset, subset_name, vocab)
            # print("\tload features and target")
            # with open(f"{subset_name}_X_y.pkl", "rb") as f:
                # X_dict = pkl.load(f)
           
            # print("\tcalculating beta")
            # calculate_beta(X_dict, vocab, subset_name)
            # print("load betas")
            # with open(f"{subset_name}_betas.pkl", "rb") as f:
            #     betas = pkl.load(f)

            
    else:
        with open("parsed_target_prime_split.pkl", 'rb') as f:
            subsets = pkl.load(f)

        names = ["mm", "mf", "fm", "ff"]
    

        for subset_name in names:
            print(subset_name)
            subset = subsets[subset_name]

            print("\tcreating vocab")
            vocab = create_vocab(subset, subset_name)
            # print("load vocab")
            # with open(f"{subset_name}_vocab.pkl", "rb") as f:
            #     vocab = pkl.load(f)

            print("\tcreating features and target")
            X_dict = create_X_y(subset, subset_name, vocab)
            # print("load c_count, y")
            # with open(f"{subset_name}_c_count_y.pkl", "rb") as f:
            #     d = pkl.load(f)
            #     c_count = d["c_count"]
            #     y = d["y"]
            # c = np.array(d["c_count"]["meetings"])
            # y = np.array(d["y"]["meetings"])
            # p = np.array(d["p_len"]["meetings"])

            # X = np.array([np.ones(len(c)), c, np.ones(len(c)), p, c, c*p, p, c*p]).T
            # glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
            # res = glm_binom.fit(maxiter=100)
            # print(res.summary())
            # print("\tcalculating beta")
            # calculate_beta(X_dict, vocab, subset_name)
            # print("load betas")
            # with open(f"{subset_name}_betas.pkl", "rb") as f:
            #     betas = pkl.load(f)
            
            # counter = 0

            # for w in betas:
            #     if betas[w][0] == "undefined":
            #         counter += 1

            # print(f"counter {counter} / {len(betas)}")
