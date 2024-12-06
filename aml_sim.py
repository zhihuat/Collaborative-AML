import pandas as pd
import numpy as np
from tqdm import *
import random
from utils import generate_hash_functions, AMLBFS, lsh, minhash_signature, BloomFilter, check_sim
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score, f1_score, roc_curve, auc

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='100K-unbalance', help="dataset name")
parser.add_argument("--rows", type=int, default=2, help="rows")
parser.add_argument("--bands", type=int, default=100, help="#bands")
args = parser.parse_args()


accounts = pd.read_csv(args.data_path + '/accounts.csv')
transactions = pd.read_csv(args.data_path + '/transactions.csv')
alerts = pd.read_csv(args.data_path + '/alert_accounts.csv')

# AML config
repeats = 5
K = 2 
len_threshold = 6 # sets larger than len_threshold will be ignored
seed = 42

# MinHash LSH config
bands = args.bands
rows = args.rows  
num_hash_functions = bands * rows

# Bloom Filter config
bf_size = 3000000
# bf_size = 500000
bf_hash_num = 7

############
alerts_list = alerts[['acct_id']].to_numpy()
transactions_list = transactions[['orig_acct', 'bene_acct']].to_numpy()

account_bank = accounts[['bank_id']].to_numpy()
bank_a = accounts.loc[np.where(account_bank == 'bank_a')[0], 'acct_id'].tolist()
bank_b = accounts.loc[np.where(account_bank == 'bank_b')[0], 'acct_id'].tolist()  
ground_truth = [1 if i in alerts_list else 0 for i in range(len(bank_a)+len(bank_b))]

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

acct_bank_dict = dict(accounts[['acct_id','bank_id']].to_numpy())  # {acct_id: bank_id}

results = []
for repeat in range(repeats):
    print('\n#### round',repeat+1, '####')
    random.seed(seed)
    seed += 1
    hash_functions = generate_hash_functions(num_hash_functions)
    
    time0 = time.time()
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

    print(f"find set A: {time.time() - time0}")
        
    ###################### Set: bank b ######################
    time0 = time.time()
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

    print(f"find set B: {time.time() - time0}")

    ###################### MinHash & BF: Bank a ######################
    time0 = time.time()
    a_forward_minhash = {key: minhash_signature(values, hash_functions) for key, values in a_forward.items()}
    a_forward_lsh = {key: lsh(signature, bands, rows) for key, signature in a_forward_minhash.items()}
    a_backward_minhash = {key: minhash_signature(values, hash_functions) for key, values in a_backward.items()}
    a_backward_lsh = {key: lsh(signature, bands, rows) for key, signature in a_backward_minhash.items()}

    print(f"minhash A: {time.time() - time0}")
    time0 = time.time()
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
    
    print(f"insert A: {time.time() - time0}")


    ###################### MinHash & BF: Bank b ######################
    time0 = time.time()
    b_forward_minhash = {key: minhash_signature(values, hash_functions) for key, values in b_forward.items()}
    b_forward_lsh = {key: lsh(signature, bands, rows) for key, signature in b_forward_minhash.items()}
    b_backward_minhash = {key: minhash_signature(values, hash_functions) for key, values in b_backward.items()}
    b_backward_lsh = {key: lsh(signature, bands, rows) for key, signature in b_backward_minhash.items()}

    print(f"minhash B: {time.time() - time0}")
    time0 = time.time()
    
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
    print(f"insert B: {time.time() - time0}")
    
    
    ###################### Comparation ######################
    time0 = time.time()
    roc_list_a_forward = np.zeros(len(accounts))
    roc_list_a_backward = np.zeros(len(accounts))
    roc_list_b_forward = np.zeros(len(accounts))
    roc_list_b_backward = np.zeros(len(accounts))


    for idx, forward in a_forward_lsh.items():
        # match_list = np.array([bf_b_backward.contains(hash_value) for hash_value in hash_list])
        match_list = np.array([bf_b_backward[k].contains(band) for (k, band) in enumerate(forward)])
        match_num = np.sum(match_list!=0)

        if match_num / bands > roc_list_a_forward[idx]:
            roc_list_a_forward[idx] = match_num / bands

    
    for idx, forward in b_forward_lsh.items():
        match_list = np.array([bf_a_backward[k].contains(band) for (k, band) in enumerate(forward)])
        match_num = np.sum(match_list!=0)

        if match_num / bands > roc_list_b_forward[idx]:
            roc_list_b_forward[idx] = match_num / bands
        

    for idx, backward in a_backward_lsh.items():
        match_list = np.array([bf_b_forward[k].contains(band) for (k, band) in enumerate(backward)])
        match_num = np.sum(match_list!=0)

        if match_num / bands > roc_list_a_backward[idx]:
            roc_list_a_backward[idx] = match_num / bands

    
    for idx, backward in b_backward_lsh.items():
        match_list = np.array([bf_a_forward[k].contains(band) for (k, band) in enumerate(backward)])
        match_num = np.sum(match_list!=0)

        if match_num / bands > roc_list_b_backward[idx]:
            roc_list_b_backward[idx] = match_num / bands
            
    print(f"compare A&B: {time.time() - time0}")


    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for threshold in thresholds:
        roc_list = np.zeros(len(accounts))
        a_forward_idx = np.where(roc_list_a_forward > threshold**rows)[0]
        a_backward_idx = np.where(roc_list_a_backward > threshold**rows)[0]
        b_forward_idx = np.where(roc_list_b_forward > threshold**rows)[0]
        b_backward_idx = np.where(roc_list_b_backward > threshold**rows)[0]
        
        for src in a_forward_idx:
            for dst in b_backward_idx:
                union_set = np.array(list(set(a_forward_account[src]) | set(b_backward_account[dst])))
                if len(union_set) > 0:
                    roc_list[union_set] = 1

        for src in b_forward_idx:
            for dst in a_backward_idx:
                union_set = np.array(list(set(b_forward_account[src]) | set(a_backward_account[dst])))
                if len(union_set) > 0:
                    roc_list[union_set] = 1
        

        infer_list = roc_list.tolist()
        
        results.append({
            'iteration': repeat,  # current iteration number
            'threshold': threshold,
            'acc': accuracy_score(ground_truth, infer_list),
            'precision': precision_score(ground_truth, infer_list),
            'recall': recall_score(ground_truth, infer_list),
            'f1': f1_score(ground_truth, infer_list),
            'auc': roc_auc_score(ground_truth, roc_list)
        })
        
results_df = pd.DataFrame(results)

final_results = results_df.groupby('threshold').agg(
    mean_accuracy=('acc', 'mean'),
    std_accuracy=('acc', 'std'),
    mean_precision=('precision', 'mean'),
    std_precision=('precision', 'std'),
    mean_recall=('recall', 'mean'),
    std_recall=('recall', 'std'),
    mean_f1=('f1', 'mean'),
    std_f1=('f1', 'std'),
    mean_auc=('auc', 'mean'),
    std_auc=('auc', 'std')
).reset_index()

final_results = final_results.round(4)
print(final_results)
save_path = 'results/' + args.data_path.split('/')[-1].split('_')[0] + f'_thred{len_threshold}_rows{rows}_bands{bands}.csv'
final_results.to_csv(save_path, index=False)

