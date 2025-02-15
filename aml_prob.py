"""
Probability-based
"""
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import random
from utils import generate_hash_functions, AMLBFS, lsh, minhash_signature, BloomFilter
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import argparse
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='/path/to/100K-balance', help="dataset name")
parser.add_argument("--rows", type=int, default=5, help="#rows")
parser.add_argument("--bands", type=int, default=100, help="#bands")
parser.add_argument("--repeats", type=int, default=1, help="repeat experiments")
parser.add_argument("--output_dir", type=str, default='results/prob', help="output dir")
parser.add_argument("--bf_size", type=int, default=500000, help="size of bloom filter")

args = parser.parse_args()


accounts = pd.read_csv(args.data_path + '/accounts.csv')
transactions = pd.read_csv(args.data_path + '/transactions.csv')
alerts = pd.read_csv(args.data_path + '/alert_accounts.csv')

save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)

# AML config
repeats = args.repeats
K = 2
seed = 42

if "100K" in args.data_path:
    len_threshold = 5 # sets smaller than len_threshold will be ignored
elif "HI-Small_Trans" in args.data_path:
    len_threshold = 8
elif "Small_Trans" in args.data_path:
    len_threshold = 9
else:
    KeyError("set 'len_threshold' for your dataset.")


# MinHash LSH config
bands = args.bands
rows = args.rows  
num_hash_functions = bands * rows

# Bloom Filter config
bf_size = args.bf_size
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
    time0 = time.time()
    a_forward = {}
    a_backward = {}
    a_forward_account = {}
    a_backward_account = {}
    

    for index, node_id in tqdm(enumerate(bank_a), desc="Set discover for band A"):
        # print(index)
        forward_set, forward_set_account = AMLBFS(src2dst, acct_bank_dict, node_id, K, reverse=False)
        backward_set, backward_set_account = AMLBFS(dst2src, acct_bank_dict, node_id, K, reverse=True)
        
        if len(forward_set) >= len_threshold:
            a_forward[node_id] = list(forward_set)
            a_forward_account[node_id] = forward_set_account
        if len(backward_set) >= len_threshold:
            a_backward[node_id] = list(backward_set)
            a_backward_account[node_id] = backward_set_account
            
    print(f"Time usage for set discovery in bank A: {time.time() - time0}")

    ###################### Set: bank b ######################
    time0 = time.time()
    b_forward = {}
    b_backward = {}
    b_forward_account = {}
    b_backward_account = {}

    
    for index, node_id in tqdm(enumerate(bank_b), desc="Set discover for band B"):
        forward_set, forward_set_account = AMLBFS(src2dst, acct_bank_dict, node_id, K, reverse=False)
        backward_set, backward_set_account = AMLBFS(dst2src, acct_bank_dict, node_id, K, reverse=True)
        if len(forward_set) >= len_threshold:
            b_forward[node_id] = list(forward_set)
            b_forward_account[node_id] = forward_set_account
        if len(backward_set) >= len_threshold:
            b_backward[node_id] = list(backward_set)
            b_backward_account[node_id] = backward_set_account    
            
    print(f"Time usage to find sets for nodes in bank B: {time.time() - time0}")
    
    ###################### MinHash & BF: Bank a ######################
    time0 = time.time()
    a_forward_minhash = {key: minhash_signature(values, hash_functions) for key, values in a_forward.items()}
    a_forward_lsh = {key: lsh(signature, bands, rows) for key, signature in a_forward_minhash.items()}
    a_backward_minhash = {key: minhash_signature(values, hash_functions) for key, values in a_backward.items()}
    a_backward_lsh = {key: lsh(signature, bands, rows) for key, signature in a_backward_minhash.items()}

    print(f"Time usage for computing minhash in bank A: {time.time() - time0}")
    
    time0 = time.time()
    # BloomFilter a
    bf_a_forward = []
    bf_a_backward = []
    for band in range(bands):
        bf1 = BloomFilter(bf_size, bf_hash_num) # Initialize BloomFilter
        bf2 = BloomFilter(bf_size, bf_hash_num) 
        for bands_list_forward in a_forward_lsh.values():
            bf1.add(bands_list_forward[band])
        
        for bands_list_backward in a_backward_lsh.values():
            bf2.add(bands_list_backward[band])
        bf_a_forward.append(bf1)
        bf_a_backward.append(bf2)

    print(f"Time usage for inserting Bloom filters in bank A: {time.time() - time0}")

    ###################### MinHash & BF: Bank b ######################
    time0 = time.time()
    b_forward_minhash = {key: minhash_signature(values, hash_functions) for key, values in b_forward.items()}
    b_forward_lsh = {key: lsh(signature, bands, rows) for key, signature in b_forward_minhash.items()}
    b_backward_minhash = {key: minhash_signature(values, hash_functions) for key, values in b_backward.items()}
    b_backward_lsh = {key: lsh(signature, bands, rows) for key, signature in b_backward_minhash.items()}

    print(f"Time usage for computing minhash in bank B: {time.time() - time0}")
    time0 = time.time()
    # BloomFilter b
    bf_b_forward = []
    bf_b_backward = []
    for band in range(bands):
        bf1 = BloomFilter(bf_size, bf_hash_num) # Initialize BloomFilter
        bf2 = BloomFilter(bf_size, bf_hash_num) 
        for bands_list_forward in b_forward_lsh.values():
            bf1.add(bands_list_forward[band]) 
        
        for bands_list_backward in b_backward_lsh.values():
            bf2.add(bands_list_backward[band])
        bf_b_forward.append(bf1)
        bf_b_backward.append(bf2)
    print(f"Time usage for inserting Bloom filters in bank B: {time.time() - time0}")
    
    ###################### Comparation ######################
    time0 = time.time()
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
            
    print(f"Time usage for membership testing: {time.time() - time0}")

    roc_list = np.zeros(len(accounts))
    a_forward_idx = np.where(roc_list_a_forward == 1)[0]
    a_backward_idx = np.where(roc_list_a_backward == 1)[0]
    b_forward_idx = np.where(roc_list_b_forward == 1)[0]
    b_backward_idx = np.where(roc_list_b_backward == 1)[0]
    
    roc_list[a_forward_idx] = 1
    roc_list[a_backward_idx] = 1
    roc_list[b_forward_idx] = 1
    roc_list[b_backward_idx] = 1
    
    # find intermediate nodes using src and dst nodes
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
    
    results.append({
        'iteration': repeat,  # current iteration number
        'acc': accuracy_score(ground_truth, infer_list),
        'precision': precision_score(ground_truth, infer_list),
        'recall': recall_score(ground_truth, infer_list),
        'f1': f1_score(ground_truth, infer_list),
    })
        
    
results_df = pd.DataFrame(results)

final_results = results_df.agg({
    "acc": ['mean', 'std'], 
    "precision": ['mean', 'std'],
    "recall": ['mean', 'std'],
    "f1": ['mean', 'std']
    })


final_results = final_results.round(4)
print(final_results)
save_path = os.path.join(save_dir, args.data_path.split('/')[-1].split('_')[0] + f'_thred{len_threshold}_rows{rows}_bands{bands}.csv')
final_results.to_csv(save_path, index=False)
