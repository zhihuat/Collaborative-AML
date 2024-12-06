import pandas as pd

def set_similarity(set_a, set_b): 
    return len(set_a & set_b) / len(set_a | set_b) if len(set_a | set_b) != 0 else 0.

def calculate_edge(a, b, reverse):
    src = int(a) if not reverse else int(b)
    dst = int(b) if not reverse else int(a)
    return (src + dst) * (src + dst + 1) // 2 + src

# def AMLBFS(graph, acct_bank_dict, s, k, reverse):
#     bank_s = acct_bank_dict[s]
#     queue = []
#     queue.append([s, k])
#     nodes_set = set()
#     edge_set =[]
#     while (len(queue) > 0):
#         vertex, hop = queue.pop(0)
#         if vertex in nodes_set:
#             continue
#         nodes = graph[vertex]
#         if hop <= 0:
#             break
#         for w in nodes:
#             if acct_bank_dict[w] != bank_s:   # k-hop的BFS，当遇到cross-client的账户时，记录进set
#                 edge_set.append(calculate_edge(vertex, w, reverse))
#                 nodes_set.add(w)
#                 nodes_set.add(vertex)
#             elif w in nodes_set:
#                 continue
#             else:
#                 queue.append([w, hop - 1])   # 否则继续在client内进行hop-1跳的传递
#     return edge_set, nodes_set

def AMLBFS(graph, acct_bank_dict, s, k, reverse):
    bank_s = acct_bank_dict[s]  # Get the bank associated with the starting node
    queue = [[s, k]]  # Initialize the queue with the starting node and remaining hops
    nodes_set = []  # Set to keep track of encountered nodes
    edge_set = []  # List to record cross-bank edges

    while queue:
        vertex, hop = queue.pop(0)  # Dequeue a node and its remaining hops
        if vertex in nodes_set:
            continue
        if hop <= 0:
            continue
        
        nodes_set.append(vertex) 
        neighbors = graph.get(vertex, [])
        
        for neighbor in neighbors:
            if acct_bank_dict[neighbor] != bank_s:  # Cross-bank condition
                # Record the cross-bank transaction
                edge_set.append(calculate_edge(vertex, neighbor, reverse))
                nodes_set.append(neighbor)  # Add neighbor to the set
            elif neighbor in nodes_set:
                continue
            elif neighbor not in nodes_set:
                queue.append([neighbor, hop - 1])  # Continue BFS with remaining hops

    return list(set(edge_set)), list(set(nodes_set))



def AMLBFS(graph, acct_bank_dict, s, k, reverse):
    bank_s = acct_bank_dict[s]  # Get the bank associated with the starting node
    queue = [[s, k]]  # Initialize the queue with the starting node and remaining hops
    nodes_set = set()  # Set to keep track of encountered nodes
    edge_set = []  # List to record cross-bank edges

    while queue:
        vertex, hop = queue.pop(0)  # Dequeue a node and its remaining hops
        if vertex in nodes_set:  # If node has already been visited, skip it
            continue
        if hop <= 0:  # If no hops left, skip further exploration
            continue
        
        nodes_set.add(vertex)  # Mark the current node as visited
        neighbors = graph.get(vertex, [])  # Get all neighbors of the current node
        
        for neighbor in neighbors:
            # If neighbor belongs to a different bank, it's a cross-bank transaction
            if acct_bank_dict[neighbor] != bank_s:
                # Record the cross-bank transaction edge
                edge_set.append(calculate_edge(vertex, neighbor, reverse))
                nodes_set.add(neighbor)  # Mark the neighbor as visited
            elif neighbor not in nodes_set:
                queue.append([neighbor, hop - 1])  # Continue BFS with remaining hops

    # Remove duplicates from edge_set and nodes_set
    return list(set(edge_set)), list(nodes_set)

import numpy as np
import pandas as pd

def find_transactions(transactions_np, accounts_np, acct_id, reverse,  K):
    # 定义计算公式
    
    
    # 将 DataFrame 转换为 NumPy 数组
    # accounts_np = accounts_df.to_numpy()
    # transactions_np = transactions_df.to_numpy()
    
    direct_transactions_mask = (transactions_np[:, 0] == acct_id)
    direct_transactions_np = transactions_np[direct_transactions_mask]
    
    # 找到这些交易的接收账户
    bene_accts = np.unique(direct_transactions_np[:, 1])
    
    # 找到这些接收账户的进一步交易
    second_level_transactions_mask = np.isin(transactions_np[:, 0], bene_accts)
    second_level_transactions_np = transactions_np[second_level_transactions_mask]
    
    all_transactions_np = np.vstack((direct_transactions_np, second_level_transactions_np))
    
    
    orig_bank_ids = accounts_np[all_transactions_np[:, 0], 1]
    bene_bank_ids = accounts_np[all_transactions_np[:, 1], 1]
    
    # filtered_mask = orig_bank_ids != bene_bank_ids
    filtered_transactions_np = all_transactions_np[orig_bank_ids != bene_bank_ids]
    
    if len(filtered_transactions_np) > 0:
        result_transactions = [calculate_edge(a, b, reverse) for a, b in filtered_transactions_np]
        related_account_ids = np.concatenate((all_transactions_np[:, 0], all_transactions_np[:, 1]))
        related_account_ids = np.unique(related_account_ids)
        other_account_ids = related_account_ids[related_account_ids != acct_id]
        
    else:
        result_transactions = []
        other_account_ids = []
        
    return result_transactions, other_account_ids


def get_transactions(account_id, transactions_dict, accounts_array, max_depth=2):
    visited_accounts = set()
    valid_transactions = []

    def dfs(current_account, current_depth):
        visited_accounts.add(current_account)
        
        if current_depth > max_depth:
            return
        
        for recipient_account in transactions_dict.get(current_account, []):
            # if recipient_account not in visited_accounts:
                origin_bank_id = accounts_array[current_account]['bank_id']
                recipient_bank_id = accounts_array[recipient_account]['bank_id']

                if origin_bank_id != recipient_bank_id:
                    # transaction_amount = transactions_dict[current_account][recipient_account]
                    
                    # Assuming 'a' is the sender's ID and 'b' is the receiver's ID
                    a = current_account
                    b = recipient_account
                    
                    transaction_value = (a + b) * (a + b + 1) / 2 + a
                    
                    valid_transactions.append({
                        # 'origin': current_account,
                        # 'destination': recipient_account,
                        # 'amount': transaction_amount,
                        'value': transaction_value,
                        # 'origin_bank': origin_bank_id,
                        # 'destination_bank': recipient_bank_id
                    })
                
                dfs(recipient_account, current_depth + 1)

    dfs(account_id, 0)
    return valid_transactions



import random
import hashlib

def generate_hash_functions(num_hash_functions):  # return [hash_func1, hash_func2, ..., hash_func]
    hash_funcs = []
    for _ in range(num_hash_functions):
        a = random.randint(1, 2**32-1)
        b = random.randint(1, 2**32-1)
        hash_funcs.append(lambda x, a=a, b=b: (a * x + b) % (2**32-1))
    return hash_funcs

def minhash_signature(document, hash_functions): # return [min hash, minhash, minhash]
    shingles = set(document) # speed up
    signature = [min(h(s) for s in shingles) for h in hash_functions]
    return signature

def jaccard_similarity(s1, s2):
    intersection = len(s1.intersection(s2))
    union = len(s1) + len(s2) - intersection
    return intersection / union

def minhash_similarity(signature1, signature2):
    equal_hashes = sum(1 for h1, h2 in zip(signature1, signature2) if h1 == h2)
    return equal_hashes / len(signature1)

def lsh(signature, bands, rows):
    band_hashes = []
    for band in range(bands):
        band_signature = signature[band * rows:(band + 1) * rows]
        hash_object = hashlib.md5(bytes(str(band_signature), 'utf-8'))
        band_hash = hash_object.hexdigest()
        band_hashes.append(band_hash)
    return band_hashes


import hashlib

class BloomFilter:
    def __init__(self, n, k):
        self.n = n
        self.k = k 
        self.bloom = [False] * n

    def add(self, item):
        for i in range(self.k):
            hash_val = int(hashlib.sha256(str(item).encode('utf-8') + str(i).encode('utf-8')).hexdigest(), 16)
            index = hash_val % self.n
            self.bloom[index] = True

    def contains(self, item):
        for i in range(self.k):
            hash_val = int(hashlib.sha256(str(item).encode('utf-8') + str(i).encode('utf-8')).hexdigest(), 16)
            index = hash_val % self.n
            if not self.bloom[index]:
                return False
        return True


def check_sim(accounts, sg, a_forward, a_backward, b_forward, b_backward):
    for idx, row in sg.iterrows():
        src, dst = row['source'], row['dest']
        
        src_id = accounts.loc[accounts['account_id']==src, 'acct_id'].values[0]
        dst_id = accounts.loc[accounts['account_id']==dst, 'acct_id'].values[0]
        
        
        sim1 = set_similarity(set(a_forward.get(src_id, [])), set(b_backward.get(dst_id, [])))
        sim2 = set_similarity(set(b_forward.get(src_id, [])), set(a_backward.get(dst_id, [])))
        
        print(max(sim1, sim2))