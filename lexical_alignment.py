import pickle as pkl
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
  
# create subsets
def create_subsets(filename="subset_dict.pkl"):

    # load parsed file
    with open("Corpora/ICSI/adjacency_list.pkl", "rb") as f:
        adj_list = pkl.load(f)
    
    subset_dict = dict(mm=list(),
                    mf=list(),
                    fm=list(),
                    ff=list())

    for item in adj_list:
        gender_a = item["a"]["gender"] 
        gender_b = item["b"]["gender"] 
        if gender_a == "m" and gender_b == "m":
            subset_name = "mm"
        elif gender_a == "m" and gender_b == "f":
            subset_name = "mf"
        elif gender_a == "f" and gender_b == "m":
            subset_name = "fm"
        else: # gneder_a == "f" and gender_b == "f"
            subset_name = "ff"

        subset_dict[subset_name].append((item["a"]["counter"], item["b"]["counter"]))

    with open(f"{filename}", "wb") as f:
        pkl.dump(subset_dict, f)

    return subset_dict


# create vocab for subset
def create_vocab(subset, subset_name):
    vocab = set()
    for p, t in subset:
        vocab |= set(p.keys())
        vocab |= set(t.keys())

    with open(f"{subset_name}_vocab.pkl", "wb") as f:
        pkl.dump(vocab, f)

    return vocab


# create the features and labels
def create_X_y(subset, subset_name, vocab):
    c_count = {w: list() for w in vocab}
    y = {w: list() for w in vocab}
    p_len = {w: list() for w in vocab}

    for w in vocab:
        for p, t in subset:
            c_count[w].append(p[w])
            y[w].append(1 if t[w] > 0 else 0)
            p_len[w].append(sum(p.values()))

    d = dict(c_count=c_count, y=y, p_len=p_len)
    with open(f"{subset_name}_c_count_y.pkl", "wb") as f:
        pkl.dump(d, f)

    return d


# calculate beta
def calculate_beta(X_dict, vocab, subset_name):
    c_count = X_dict["c_count"]
    y = X_dict["y"]
    p_len = X_dict["p_len"]

    beta_dict = {w: {i: "undefined" for i in range(7)} for w in vocab}

    for w in vocab:
        c_w = c_count[w]
        y_w = y[w]
        p_len_w = p_len[w]

        if sum(y_w) > 0:
            c_w = np.array(c_w)
            p_len_w = np.array(p_len_w)
            X = np.array([c_w, np.ones(len(c_w)), p_len_w, c_w, c_w*p_len_w, p_len_w, c_w*p_len_w]).T
            y_w = np.array(y_w)

            clf = LogisticRegression(random_state=0).fit(X, y_w)
            beta_dict[w][0] = clf.intercept_[0]
            coef = clf.coef_.squeeze()
            for i, c in enumerate(coef):
                beta_dict[w][i+1] = c
                

    with open(f"{subset_name}_betas.pkl", "wb") as f:
        pkl.dump(beta_dict, f)


if __name__ == "__main__":
    print("load subsets\n")
    with open("subset_dict.pkl", "rb") as f:
        subsets = pkl.load(f)

    # names = ["fm", "ff"]
    names = ["ff"]
    for subset_name in names:
        print(subset_name)
        subset = subsets[subset_name]

        # print("\tcreating vocab")
        # vocab = create_vocab(subset, subset_name)
        # print("load vocab")
        # with open(f"{subset_name}_vocab.pkl", "rb") as f:
        #     vocab = pkl.load(f)

        # print("\tcreating c_count and y")
        # X_dict = create_X_y(subset, subset_name, vocab)
        print("load c_count, y")
        with open(f"{subset_name}_c_count_y.pkl", "rb") as f:
            d = pkl.load(f)
        #     c_count = d["c_count"]
        #     y = d["y"]
        c = np.array(d["c_count"]["meetings"])
        y = np.array(d["y"]["meetings"])
        p = np.array(d["p_len"]["meetings"])

        X = np.array([np.ones(len(c)), c, np.ones(len(c)), p, c, c*p, p, c*p]).T
        glm_binom = sm.GLM(y, X, family=sm.families.Binomial())
        res = glm_binom.fit(maxiter=100)
        print(res.summary())
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
