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

    if ARGS.analysis == 2:
        return pf_apl, pm_apl

    apl = pf_apl + pm_apl

    with open(f"{SAVEDIR}/t_{target_gender}_prepped_apl.pkl", "wb") as f:
        pkl.dump(apl, f)

    return apl


def create_predictors(apl):
    print("Creating predictors...")

    y = [ 1 if target > 0 else 0 for _, _, target, _ in apl]

    if ARGS.analysis == 1:
        c_count = [prime for _, prime, _, _ in apl]
        c_gender = [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl]
        c_plen = [plen for _, _, _, plen in apl]

        return c_count, c_gender, c_plen, y

    else:
        c_count = [prime for _, prime, _, _ in apl]

        return c_count, y        


def calculate_beta(apl):
    if ARGS.analysis == 1:
        n = 8
        c_count, c_gender, c_plen, y = create_predictors(apl)
    else:
        n = 2
        c_count, y = create_predictors(apl)

    pvalue_dict = {i: "undefined" for i in range(n)}
    zscores_dict = {i: "undefined" for i in range(n)}
    betas_dict = {i: "undefined" for i in range(n)}

    print("Calculating betas...")
    if sum(y) > 0:
        c_w = np.array(c_count)
        
        if ARGS.analysis == 1:
            g_w = np.array(c_gender)
            X = np.array([np.ones(len(c_w)), c_w, g_w, c_plen, c_w*g_w, c_w*c_plen, g_w*c_plen, c_w*g_w*c_plen]).T
        else:
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


def calculate_alignment(apl, target_gender, prime_gender="f"):
    print(f"Calculating alignment for experiment {ARGS.analysis}, ",  
        f"t_{target_gender}" if ARGS.analysis == 1 else f"p_{prime_gender}_t_{target_gender}")

    z, p, b = calculate_beta(apl)

    dir_name = f"./lexical_{ARGS.dataset}"
    au.print_results(z, p, b, dir_name, ARGS.between, ARGS.analysis, target_gender, prime_gender)


def main():
    au.prep_dataset(ARGS.dataset, between=ARGS.between, lexical=True)

    os.makedirs(SAVEDIR, exist_ok=True)

    for target_gender in ["m", "f"]:

        if ARGS.analysis == 1:
            adj_pair_list = prep_apl(target_gender)
            
            calculate_alignment(adj_pair_list, target_gender)

        else:
            female_prime_apl, male_prime_apl = prep_apl(target_gender)

            calculate_alignment(female_prime_apl, target_gender)

            calculate_alignment(male_prime_apl, target_gender, prime_gender="m")

    print("Done! Enjoy the results!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for lexical alignment.')

    parser.add_argument('dataset', type=str,
						help="\"AMI\" or \"ICSI\", name of the dataset being analysed.")
    parser.add_argument('analysis', type=int, default=1,
						help="1 or 2, experiment to perform")
    parser.add_argument('--between', type=bool, default=False,
                        help="bool to include the intermiediate adjacency pairs or not, default False")


    ARGS = parser.parse_args()

    print("Attention: Make sure you're in the 'Alignment' directory before running code!")
    print(ARGS)

    SUBDIR = "between" if ARGS.between else "plain"
    SAVEDIR = f"./lexical_{ARGS.dataset}/{SUBDIR}"

    main()