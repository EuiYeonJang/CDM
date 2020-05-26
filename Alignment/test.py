import pickle as pkl
import numpy as np
import os
import statsmodels.api as sm
import argparse
import more_itertools.more

from collections import Counter
import alignment_utils as au


def prep_apl(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between, lexical=True)

    apl = pf_apl + pm_apl

    with open(f"{SAVEDIR}/t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(apl, f)

    return apl


def prep_apl_two(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between, lexical=True)

    return pf_apl, pm_apl


def create_predictors(apl, eq):
    print("Creating predictors...")
    y = [ 1 if target > 0 else 0 for _, _, target, _ in apl]

    if eq == 1:
        c_count = [prime for _, prime, _, _ in apl]
        c_gender = [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl]

        return c_count, c_gender, y
    
    elif eq == 2:
        c_count = [prime for _, prime, _, _ in apl]

        return c_count, y

    elif eq == 3: 
        c_count = [prime for _, prime, _, _ in apl]
        c_gender = [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl]
        c_plen = [plen for _, _, _, plen in apl]

        return c_count, c_gender, c_plen, y


def calculate_beta_one(apl):
    c_count, c_gender, y = create_predictors(apl, 1)

    print("Calculating betas...")
    pvalue_dict = {i: "undefined" for i in range(4)}
    zscores_dict = {i: "undefined" for i in range(4)}
    betas_dict = {i: "undefined" for i in range(4)}

    if sum(y) > 0:
        c_w = np.array(c_count)
        g_w = np.array(c_gender)
        X = np.array([np.ones(len(c_w)), c_w, g_w, c_w*g_w]).T

        y_w = np.array(y)


        res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
        
        p_values = res.pvalues
        for i, p in enumerate(p_values):
            pvalue_dict[i] = p

        z_scores = res.tvalues
        for i, z in enumerate(z_scores):
            zscores_dict[i] = z

        betas = res.params
        for i, b in enumerate(betas):
            betas_dict[i] = b
    
    return zscores_dict, pvalue_dict, betas_dict


def calculate_beta_two(apl):
    c_count, y = create_predictors(apl, 2)
    
    pvalue_dict = {i: "undefined" for i in range(2)}
    zscores_dict = {i: "undefined" for i in range(2)}
    betas_dict = {i: "undefined" for i in range(2)}

    if sum(y) > 0:
        c_w = np.array(c_count)
        X = np.array([np.ones(len(c_w)), c_w]).T

        y_w = np.array(y)

        res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
        
        p_values = res.pvalues
        for i, p in enumerate(p_values):
            pvalue_dict[i] = p

        z_scores = res.tvalues
        for i, z in enumerate(z_scores):
            zscores_dict[i] = z

        betas = res.params
        for i, b in enumerate(betas):
            betas_dict[i] = b

    return zscores_dict, pvalue_dict, betas_dict


def calculate_beta_three(apl):
    c_count, c_gender, c_plen, y = create_predictors(apl, 3)

    pvalue_dict = {i: "undefined" for i in range(8)}
    zscores_dict = {i: "undefined" for i in range(8)}
    betas_dict = {i: "undefined" for i in range(8)}


    if sum(y) > 0:
        c_w = np.array(c_count)
        g_w = np.array(c_gender)
        X = np.array([np.ones(len(c_w)), c_w, g_w, c_plen, c_w*g_w, c_w*c_plen, g_w*c_plen, c_w*g_w*c_plen]).T

        y_w = np.array(y)

        res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
        
        p_values = res.pvalues
        for i, p in enumerate(p_values):
            pvalue_dict[i] = p

        z_scores = res.tvalues
        for i, z in enumerate(z_scores):
            zscores_dict[i] = z

        betas = res.params
        for i, b in enumerate(betas):
            betas_dict[i] = b

    return zscores_dict, pvalue_dict, betas_dict


def calculate_alignment(apl, eq, target_gender, prime="f"):
    print(f"Calculating alignment for eq {eq}, p_{prime}_t_{target_gender}")
    betas = [0, 1, 2, 3] if eq == 1 else [0, 1] if eq == 2 else [0, 1, 2, 3, 4, 5, 6, 7]

    if eq == 1:
        z, p, b = calculate_beta_one(apl)
    elif eq == 2:
        z, p, b = calculate_beta_two(apl)
    else:
        z, p, b = calculate_beta_three(apl)

    results_filename = f"./lexical_{ARGS.dataset}/results_between.txt" if ARGS.between else f"./lexical_{ARGS.dataset}/results_orig.txt"

    with open(results_filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EQUATION: {ARGS.analysis}\nTARGET GENDER: {target_gender}\n")

        if eq == 2: f.write(f"PRIME GENDER: {prime}\n")

        f.write("==================================\n")

        f.write("BETA:\tZ-SCORES\tP-VALUES\n")
        f.write("----------------------------------\n")

        for bb in betas:
            if p[bb] < 0.05: f.write(">> ")
            f.write(f"{bb}\t{z[bb]:.3f}\t{p[bb]:.3f}\n")


    betas_filename = f"./lexical_{ARGS.dataset}/betas_between.txt" if ARGS.between else f"./lexical_{ARGS.dataset}/betas_orig.txt"

    with open(betas_filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EQUATION: {ARGS.analysis}\nTARGET GENDER: {target_gender}\n")

        if eq == 2: f.write(f"PRIME GENDER: {prime}\n")

        f.write("==================================\n")
        betas = [0, 1, 2, 3] if eq == 1 else [0, 1] if eq == 2 else [0, 1, 2, 3, 4, 5, 6, 7]

        for bb in betas:
            f.write(f"\tBETA_{bb}")

        f.write("\n----------------------------------\n")

        for bb in betas:
            f.write(f"\t{b[bb]:.3f}")
        f.write(f"\n")


def main():
    au.prep(ARGS.dataset, lexical=True)

    os.makedirs(SAVEDIR, exist_ok=True)
    for target_gender in ["m", "f"]:

        if ARGS.analysis == 1 or ARGS.analysis == 3:
            filename = f"{SAVEDIR}/t_{target_gender}_prepped_apl.pkl"

            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    adj_pair_list = pkl.load(f)

            else:
                adj_pair_list = prep_apl(target_gender)

            calculate_alignment(adj_pair_list, ARGS.analysis, target_gender)

        elif ARGS.analysis == 2:
            female_prime_apl, male_prime_apl = prep_apl_two(target_gender)

            calculate_alignment(female_prime_apl, ARGS.analysis, target_gender)

            calculate_alignment(male_prime_apl, ARGS.analysis, target_gender, prime="m")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for lexical alignment.')
    parser.add_argument('--dataset', type=str, default="AMI",
						help="\"AMI\" or \"ICSI\", name of the dataset being analysed.")
    parser.add_argument('--analysis', type=int, default=1,
						help="1, 2 or 3, the type of analysis to perform (equation number)")
    parser.add_argument('--between', type=bool, default=False,
                        help="bool to include the intermiediate utterances or not, default True")


    ARGS = parser.parse_args()

    print("Attention: Make sure you're in the 'Alignment' directory before running code!")
    print(ARGS)

    SAVEDIR = f"./lexical_{ARGS.dataset}/between" if ARGS.between else f"./lexical_{ARGS.dataset}/orig"

    main()