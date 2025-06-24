"""
Probability based
"""
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import *
import random
from utils import generate_hash_functions, AMLBFS, lsh, minhash_signature, BloomFilter, find_transactions, check_sim
import argparse
import pickle
import os


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='100K-unbalance', help="dataset name")
parser.add_argument("--rows", type=int, default=5, help="#rows")
parser.add_argument("--bands", type=int, default=60, help="#bands")
args = parser.parse_args()


accounts = pd.read_csv(args.data_path + '/accounts.csv')
transactions = pd.read_csv(args.data_path + '/transactions.csv')
alerts = pd.read_csv(args.data_path + '/alert_accounts.csv')


# AML config
repeats = 5

K = 2
len_threshold = 7  # sets larger than len_threshold will be ignored
seed = 42

# MinHash LSH config
bands = args.bands
rows = args.rows  
num_hash_functions = bands * rows

# Bloom Filter config
bf_size = 3000000
bf_hash_num = 7


alerts_list = alerts[['acct_id']].to_numpy()
transactions_list = transactions[['orig_acct', 'bene_acct']].to_numpy()

account_bank = accounts[['bank_id']].to_numpy()
bank_a = accounts.loc[np.where(account_bank == 'bank_a')[0], 'acct_id'].tolist()
bank_b = accounts.loc[np.where(account_bank == 'bank_b')[0], 'acct_id'].tolist()  

ground_truth = [1 if i in alerts_list else 0 for i in range(len(accounts))]
# print(ground_truth)

src2dst = {}
dst2src = {}

src_node = transactions_list[:, 0]
dst_node = transactions_list[:, 1]

for acct in accounts['acct_id'].to_numpy():
    src2dst[acct] = []
    dst2src[acct] = []

for src, dst in transactions_list: 
    src2dst[src].append(dst)
    dst2src[dst].append(src)

acct_bank_dict = dict(accounts[['acct_id','bank_id']].to_numpy())  


results = []
for repeat in range(repeats):
    print('\n#### round',repeat+1, '####')
    random.seed(seed)
    seed += 1
    hash_functions = generate_hash_functions(num_hash_functions)

    ###################### Set: Bank a ######################
    a_forward = {}
    a_backward = {}
    a_forward_account = {}
    a_backward_account = {}
    

    for index, node_id in tqdm(enumerate(bank_a)):
        # print(index)
        forward_set, forward_set_account = AMLBFS(src2dst, acct_bank_dict, node_id, K, reverse=False)
        backward_set, backward_set_account = AMLBFS(dst2src, acct_bank_dict, node_id, K, reverse=True)
        
        if len(forward_set) >= len_threshold:
            a_forward[node_id] = list(forward_set)
            a_forward_account[node_id] = forward_set_account
        if len(backward_set) >= len_threshold:
            a_backward[node_id] = list(backward_set)
            a_backward_account[node_id] = backward_set_account

    ###################### Set: bank b ######################
    b_forward = {}
    b_backward = {}
    b_forward_account = {}
    b_backward_account = {}

    
    for index, node_id in tqdm(enumerate(bank_b)):
        forward_set, forward_set_account = AMLBFS(src2dst, acct_bank_dict, node_id, K, reverse=False)
        backward_set, backward_set_account = AMLBFS(dst2src, acct_bank_dict, node_id, K, reverse=True)
        if len(forward_set) >= len_threshold:
            b_forward[node_id] = list(forward_set)
            b_forward_account[node_id] = forward_set_account
        if len(backward_set) >= len_threshold:
            b_backward[node_id] = list(backward_set)
            b_backward_account[node_id] = backward_set_account    
    
    ###################### MinHash & BF: Bank a ######################
    a_forward_minhash = {key: minhash_signature(values, hash_functions) for key, values in a_forward.items()}
    a_forward_lsh = {key: lsh(signature, bands, rows) for key, signature in a_forward_minhash.items()}
    a_backward_minhash = {key: minhash_signature(values, hash_functions) for key, values in a_backward.items()}
    a_backward_lsh = {key: lsh(signature, bands, rows) for key, signature in a_backward_minhash.items()}

    # BloomFilter a
    bf_a_forward = []
    bf_a_backward = []
    for band in range(bands):
        bf1 = BloomFilter(bf_size, bf_hash_num) # Initialize BloomFilter
        bf2 = BloomFilter(bf_size, bf_hash_num) 
        for (bands_list_forward, bands_list_backward) in tqdm(zip(a_forward_lsh.values(), a_backward_lsh.values())):
            bf1.add(bands_list_forward[band]) 
            bf2.add(bands_list_backward[band])
        bf_a_forward.append(bf1)
        bf_a_backward.append(bf2)


    ###################### MinHash & BF: Bank b ######################
    b_forward_minhash = {key: minhash_signature(values, hash_functions) for key, values in b_forward.items()}
    b_forward_lsh = {key: lsh(signature, bands, rows) for key, signature in b_forward_minhash.items()}
    b_backward_minhash = {key: minhash_signature(values, hash_functions) for key, values in b_backward.items()}
    b_backward_lsh = {key: lsh(signature, bands, rows) for key, signature in b_backward_minhash.items()}

    # BloomFilter b
    bf_b_forward = []
    bf_b_backward = []
    for band in range(bands):
        bf1 = BloomFilter(bf_size, bf_hash_num) # Initialize BloomFilter
        bf2 = BloomFilter(bf_size, bf_hash_num) 
        for (bands_list_forward, bands_list_backward) in tqdm(zip(b_forward_lsh.values(), b_backward_lsh.values())):
            bf1.add(bands_list_forward[band]) 
            bf2.add(bands_list_backward[band])
        bf_b_forward.append(bf1)
        bf_b_backward.append(bf2)
    
    ###################### Comparation ######################
    roc_list_a_forward = np.zeros(len(accounts))
    roc_list_a_backward = np.zeros(len(accounts))
    roc_list_b_forward = np.zeros(len(accounts))
    roc_list_b_backward = np.zeros(len(accounts))

    for idx, forward in a_forward_lsh.items():
        bf_lsh_match = any(bf_b_backward[k].contains(band) for (k, band) in enumerate(forward))
        if bf_lsh_match:
            roc_list_a_forward[idx] = 1


    
    for idx, forward in b_forward_lsh.items():
        bf_lsh_match = any(bf_a_backward[k].contains(band) for (k, band) in enumerate(forward))
        if bf_lsh_match:
            roc_list_b_forward[idx] = 1



    for idx, backward in a_backward_lsh.items():
        bf_lsh_match = any(bf_b_forward[k].contains(band) for (k, band) in enumerate(backward))
        if bf_lsh_match:
            roc_list_a_backward[idx] = 1


    
    for idx, backward in b_backward_lsh.items():
        bf_lsh_match = any(bf_a_forward[k].contains(band) for (k, band) in enumerate(backward))
        if bf_lsh_match:
            roc_list_b_backward[idx] = 1

    
    roc_list = np.zeros(len(accounts))
    a_forward_idx = np.where(roc_list_a_forward == 1)[0]
    a_backward_idx = np.where(roc_list_a_backward == 1)[0]
    b_forward_idx = np.where(roc_list_b_forward == 1)[0]
    b_backward_idx = np.where(roc_list_b_backward == 1)[0]
    

    for src in a_forward_idx:
        for dst in b_backward_idx:
            union_set = np.array(list(set(a_forward_account[src]) & set(b_backward_account[dst]))) # & for AMLWorld
            if len(union_set) > 0:
                roc_list[union_set] = 1

    for src in b_forward_idx:
        for dst in a_backward_idx:
            union_set = np.array(list(set(b_forward_account[src]) & set(a_backward_account[dst]))) # & for AMLWorld
            if len(union_set) > 0:
                roc_list[union_set] = 1
                
    infer_list = roc_list.tolist()
    

results_df = pd.DataFrame(results)

final_results = results_df.agg({
    "acc": ['mean', 'std'], 
    "precision": ['mean', 'std'],
    "recall": ['mean', 'std'],
    "f1": ['mean', 'std']
    })


final_results = final_results.round(4)
print(final_results)
save_path = 'results/prob/' + args.data_path.split('/')[-1].split('_')[0] + f'_thred{len_threshold}_rows{rows}_bands{bands}.csv'
final_results.to_csv(save_path, index=False)
