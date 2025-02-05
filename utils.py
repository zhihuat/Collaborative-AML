from collections import deque
import random
import hashlib

def set_similarity(set_a, set_b): 
    return len(set_a & set_b) / len(set_a | set_b) if len(set_a | set_b) != 0 else 0.

def calculate_edge(a, b, reverse):
    src = int(a) if not reverse else int(b)
    dst = int(b) if not reverse else int(a)
    return (src + dst) * (src + dst + 1) // 2 + src

def AMLBFS(graph, acct_bank_dict, s, k, reverse):
    edge_set = set()
    nodes_set = set()
    
    visited = set()
    
    queue = deque()
    queue.append((s, 0))
    
    visited.add(s)
    
    while queue:
        current_node, depth = queue.popleft()
        
    
        if depth >= k:
            continue
        
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                if acct_bank_dict.get(current_node) != acct_bank_dict.get(neighbor):
                    edge_set.add(calculate_edge(current_node, neighbor, reverse))
                    nodes_set.add(neighbor)
                    nodes_set.add(current_node)
                else:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
    
    return list(edge_set), list(nodes_set)


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