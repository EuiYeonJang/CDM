import pickle as pkl
import numpy as np
from alignment_utils import LexicalAlignmentOne
import re
from collections import Counter
import statsmodels.api as sm


##################################################################
# with open("Corpora/ICSI/in_between_split.pkl", "rb") as f:
#     mf, mm, fm, ff = pkl.load(f)

# def prep_list(female_prime_list, male_prime_list):
#     adj_pair_list = list()

#     for pair in female_prime_list:
#         combined = pair["a"] + pair["fb"]
#         if sum(combined.values()) > 0 and sum(pair["b"].values()) > 0:
#             adj_pair_list.append(("f", combined , pair["b"]))


#     for pair in male_prime_list:
#         combined = pair["a"] + pair["mb"]
#         if sum(combined.values()) > 0 and sum(pair["b"].values()) > 0:
#             adj_pair_list.append(("m",  combined, pair["b"]))

#     return adj_pair_list

# apl = prep_list(fm, mm)

# with open("./stylistic/male_apl.pkl", "wb") as f:
#     pkl.dump(apl, f)

# apl = prep_list(ff, mf)

# with open("./stylistic/female_apl.pkl", "wb") as f:
#     pkl.dump(apl, f)

######################################################
# def create_vocab(adj_pair_list):
#     vocab = set()

#     for _, prime, target in adj_pair_list:
#         vocab |= set(prime.keys())
#         vocab |= set(target.keys())

#     print(len(vocab))
#     return vocab


# with open("./stylistic/female_apl.pkl", "rb") as f:
#     adj_pair_list = pkl.load(f)


# vocab = create_vocab(adj_pair_list)

# with open("./stylistic/female_vocab.pkl", "wb") as f:
#     pkl.dump(vocab, f)


# with open("./stylistic/male_apl.pkl", "rb") as f:
#     adj_pair_list = pkl.load(f)


# vocab = create_vocab(adj_pair_list)

# with open("./stylistic/male_vocab.pkl", "wb") as f:
#     pkl.dump(vocab, f)
##########################################################

# with open("./stylistic/female_vocab.pkl", "rb") as f:
#     vocab = pkl.load(f)

# with open("liwc.pkl", "rb") as f:
#     liwc = pkl.load(f)

# for cat in liwc:
#     extension = list()
#     asterisk = set()
    
#     for w in liwc[cat]:
#         if "*" in w:
#             asterisk.add(w)
#             extension.extend([vw for vw in vocab if (w == vw or vw.startswith(w[:-1]))])
#     liwc[cat] |= (set(extension))
#     liwc[cat] -= asterisk

# with open("./stylistic/female_liwc.pkl", "wb") as f:
#     pkl.dump(liwc, f)


# with open("./stylistic/male_vocab.pkl", "rb") as f:
#     vocab = pkl.load(f)

# with open("liwc.pkl", "rb") as f:
#     liwc = pkl.load(f)

# for cat in liwc:
#     extension = list()
#     asterisk = set()
    
#     for w in liwc[cat]:
#         if "*" in w:
#             asterisk.add(w)
#             extension.extend([vw for vw in vocab if (w == vw or vw.startswith(w[:-1]))])
#     liwc[cat] |= (set(extension))
#     liwc[cat] -= asterisk

# with open("./stylistic/male_liwc.pkl", "wb") as f:
#     pkl.dump(liwc, f)
###################################################################

# with open("./stylistic/male_liwc.pkl", "rb") as f:
#     liwc = pkl.load(f)

# with open("./stylistic/male_apl.pkl", "rb") as f:
#     apl = pkl.load(f)

# inv_ind = dict()

# for cat in liwc:
#     inv_ind.update({w: cat for w in liwc[cat]})

# new_apl = list()

# for g, p, t in apl:
#     new_p = Counter([inv_ind[w] for w in p.elements() if w in inv_ind])
#     new_t = Counter([inv_ind[w] for w in t.elements() if w in inv_ind])
#     plen = sum(p.values())
#     new_apl.append((g, new_p, new_t, plen))

# with open("./stylistic/male_new_apl.pkl", "wb") as f:
#     pkl.dump(new_apl, f)


# with open("./stylistic/female_liwc.pkl", "rb") as f:
#     liwc = pkl.load(f)

# with open("./stylistic/female_apl.pkl", "rb") as f:
#     apl = pkl.load(f)

# inv_ind = dict()

# for cat in liwc:
#     inv_ind.update({w: cat for w in liwc[cat]})

# new_apl = list()

# for g, p, t in apl:
#     new_p = Counter([inv_ind[w] for w in p.elements() if w in inv_ind])
#     new_t = Counter([inv_ind[w] for w in t.elements() if w in inv_ind])
#     plen = sum(p.values())
#     new_apl.append((g, new_p, new_t, plen))

# with open("./stylistic/female_new_apl.pkl", "wb") as f:
#     pkl.dump(new_apl, f)

###################################################################

# def create_X_y(vocab, adj_pair_list):
#     c_count = {w: list() for w in vocab}
#     c_gender = {w: list() for w in vocab}
#     y = {w: list() for w in vocab}

#     for w in vocab:
#         c_count[w] = [prime[w] for _, prime, _, _ in adj_pair_list]
#         c_gender[w] = [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in adj_pair_list]
#         y[w] = [ 1 if target[w] > 0 else 0 for _, _, target, _ in adj_pair_list]

#     return dict(c_count=c_count, c_gender=c_gender, y=y)


# vocab = ['articles', 'pronoun', 'prepositions', 'negations', 'tentative', 'certainty', 'discrepancy', 'exclusive', 'inclusive']
# with open("./stylistic/male_new_apl.pkl", "rb") as f:
#     apl = pkl.load(f)

# d = create_X_y(vocab, apl)

# with open("./stylistic/male_X_y.pkl", "wb") as f:
#     pkl.dump(d, f)

# with open("./stylistic/female_new_apl.pkl", "rb") as f:
#     apl = pkl.load(f)

# d = create_X_y(vocab, apl)

# with open("./stylistic/female_X_y.pkl", "wb") as f:
#     pkl.dump(d, f)

###################################################################

# def calculate_beta(target_name, vocab, c_count, c_gender, y):
#     pvalue_dict = {w: {i: "undefined" for i in range(4)} for w in vocab}
#     zscores_dict = {w: {i: "undefined" for i in range(4)} for w in vocab}

#     for w in vocab:
#         c_w = c_count[w]
#         g_w = c_gender[w]
#         y_w = y[w]

#         if sum(y_w) > 0:
#             c_w = np.array(c_w)
#             g_w = np.array(g_w)
#             X = np.array([np.ones(len(c_w)), c_w, g_w, c_w*g_w]).T

#             y_w = np.array(y_w)

#             res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
            
#             p_values = res.pvalues
#             for i, p in enumerate(p_values):
#                 pvalue_dict[w][i] = p

#             z_scores = res.tvalues
#             for i, z in enumerate(z_scores):
#                 zscores_dict[w][i] = z
                

#     with open(f"{target_name}_pvalues.pkl", "wb") as f:
#         pkl.dump(pvalue_dict, f)

#     with open(f"{target_name}_zscores.pkl", "wb") as f:
#         pkl.dump(zscores_dict, f)


# vocab = ['articles', 'pronoun', 'prepositions', 'negations', 'tentative', 'certainty', 'discrepancy', 'exclusive', 'inclusive']
# with open("./stylistic/male_X_y.pkl", "rb") as f:
#     d = pkl.load(f)

# calculate_beta("male", vocab, d["c_count"], d["c_gender"], d["y"])

# with open("./stylistic/female_X_y.pkl", "rb") as f:
#     d = pkl.load(f)

# calculate_beta("female", vocab, d["c_count"], d["c_gender"], d["y"])

###################################################

with open(f"./stylistic/male_pvalues.pkl", "rb") as f:
    p = pkl.load(f)

with open(f"./stylistic/male_zscores.pkl", "rb") as f:
    z = pkl.load(f)

print("MALE")
for cat in p:
    print(f"{cat}: {z[cat][3]:.2f}    {p[cat][3]:.3f}")

with open(f"./stylistic/female_pvalues.pkl", "rb") as f:
    p = pkl.load(f)

with open(f"./stylistic/female_zscores.pkl", "rb") as f:
    z = pkl.load(f)


print("\nFEMALE")
for cat in p:
    print(f"{cat}: {z[cat][3]:.2f}    {p[cat][3]:.3f}")