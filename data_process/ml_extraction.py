"""
1. Extract transactions from txt files.
2. Filter transactions according to Pattern type and "Is Laundering" attribute.
3. Splits transactions into three parts, including accounts, transactions, and laundering accounts.
"""

import pandas as pd
import numpy as np
import re
import os
import copy
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/AMLWorld/HI-Small_Trans.csv", help="dataset name")
parser.add_argument("--pattern_path", type=str, default='data/AMLWorld/HI-Small_Patterns.txt', help="dataset name")
args = parser.parse_args()

# Open and read the file

# Initialize an empty list to store the data
data = []

with open(args.pattern_path, 'r') as file:
    lines = file.readlines()
######################################################################
############################### Extraction ###########################
######################################################################

current_pattern = None
# first_last_transactions = []
first_last_accounts = {"source": [], "dest":[]}
ml_accounts = []
for line in lines:
    # Check for the start of a laundering attempt
    start_match = re.match(r"BEGIN LAUNDERING ATTEMPT - (.*)", line)
    if start_match:
        current_pattern = start_match.group(1)
        first_transaction = None  # Reset for the new pattern
        last_transaction = None
        continue
    
    # Check for the end of a laundering attempt
    if "END LAUNDERING ATTEMPT" in line:
        if current_pattern == 'SCATTER-GATHER':
            if first_transaction:    
                first_last_accounts["source"].append(first_transaction[2])
            if last_transaction:
                # # There may exist Scatter-gather groups with only two accouts send money to each other
                # if last_transaction[4] in first_last_accounts["source"] or last_transaction[4] in first_last_accounts["dest"]:
                #     first_last_accounts["dest"].append(last_transaction[2])
                # else:
                first_last_accounts["dest"].append(last_transaction[4])
            
        current_pattern = None
        continue
    
    # If inside a laundering attempt, capture the transaction
    if current_pattern and re.match(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2},.*", line):
        transaction_values = line.strip().split(',')
        transaction_values.append(current_pattern)  # Add the pattern as the last column
        if current_pattern == 'SCATTER-GATHER':
            ml_accounts.append(transaction_values[2])
            ml_accounts.append(transaction_values[4])
        
        if (transaction_values[2] == transaction_values[4]) and (transaction_values[1] == transaction_values[3]): ## 'Account' == 'Account.1' and 'From Bank' == 'To Bank'
            continue
        
        data.append(transaction_values)
        
        if first_transaction is None:
            first_transaction = transaction_values
        
        # Always update the last transaction
        last_transaction = transaction_values

columns = [
    'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 
    'Amount Received', 'Receiving Currency', 'Amount Paid', 
    'Payment Currency', 'Payment Format', 'Is Laundering', 'Pattern'
]

laundering_data = pd.DataFrame(data, columns=columns)
laundering_data.to_csv(args.data_path.split('.')[0] + '_Pattern.csv', index=False)


first_last_accounts = pd.DataFrame({"source": first_last_accounts['source'], "dest": first_last_accounts['dest']})
first_last_accounts.to_csv(args.data_path.split('.')[0] + '/ScatterGather.csv', index=False)

######################################################################
############################### Filter ###############################
######################################################################

data = pd.read_csv(args.data_path)
data = data[~((data['Account'] == data['Account.1']) & (data['From Bank'] == data['To Bank']))] # Drop self transactions

merge_columns = [
    'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 
    'Amount Received', 'Receiving Currency', 'Amount Paid', 
    'Payment Currency', 'Payment Format', 'Is Laundering'
]

for column in merge_columns:
    laundering_data[column] = laundering_data[column].astype(data[column].dtype) # type alignment

df_merged = pd.merge(data, laundering_data[['Pattern'] + merge_columns],  # Merge pattern
                     on=merge_columns, how='left')
df_merged['Pattern'] = df_merged['Pattern'].where(pd.notna(df_merged['Pattern']), 'Clean')


df_sg = df_merged[
    (df_merged['Pattern'] == 'SCATTER-GATHER') # | (df_merged['Pattern'] == 'Clean') # Extract SCATTER-GATHER subgroups
]
# df_merged.loc[df_merged['Pattern'] == 'Clean', 'Is Laundering'] = 0
df_clean = df_merged[df_merged['Is Laundering'] == 0]

############### For debug, only preserve little clean transactions ###################
# random_row = df_clean.sample(n=int(0.9 * len(df_clean)))
# df_clean = df_clean.drop(random_row.index)
######################################################################################


df_filtered = pd.concat([df_sg, df_clean], ignore_index=True)

######################################################################
############################### Split ################################
######################################################################
print("Start split data")
dir = args.data_path.split('.')[0]
os.makedirs(dir, exist_ok=True)

# Assign accounts into two banks
from_bank_df = df_filtered[['Account', 'From Bank']].copy()
from_bank_df.columns = ['account_id', 'bank_id']

to_bank_df = df_filtered[['Account.1', 'To Bank']].copy()
to_bank_df.columns = ['account_id', 'bank_id']
accounts_df = pd.concat([from_bank_df, to_bank_df], ignore_index=True).drop_duplicates(ignore_index=True)

accounts_df.loc[:,'bank_id_int'] =  np.random.choice([0, 1], size=len(accounts_df))

for a, b in zip(first_last_accounts.loc[:, "source"], first_last_accounts.loc[:, "dest"]):
    accounts_df.loc[accounts_df['account_id'] == a, 'bank_id_int'] = 1 - accounts_df.loc[accounts_df['account_id'] == b, 'bank_id_int'].values[0]

accounts_df.loc[:, 'final_bank_id'] = accounts_df.loc[:, 'bank_id_int'].replace({0: 'bank_a', 1: 'bank_b'})  # Bank assignment
accounts_df['final_account_id'] = range(len(accounts_df))


# Apply the final_account_id back to the original transaction DataFrame
data_merged_accounts = df_filtered.merge(accounts_df[['account_id', 'bank_id','final_bank_id', 'final_account_id']], 
                         left_on=['Account', 'From Bank'], 
                         right_on=['account_id', "bank_id"], 
                         how='left')

data_merged_accounts.rename(columns={'final_account_id': 'Account_final_id'}, inplace=True)
data_merged_accounts.rename(columns={'final_bank_id': 'From Bank'}, inplace=True)
data_merged_accounts.drop(['account_id', 'bank_id'], axis=1, inplace=True)

# Merge the 'final_account_id' with the original 'Account.1' column
data_merged_accounts = data_merged_accounts.merge(accounts_df[['account_id', 'bank_id', 'final_bank_id', 'final_account_id']], 
                                left_on=['Account.1', 'To Bank'], 
                                right_on=['account_id', 'bank_id'], 
                                how='left')

data_merged_accounts.rename(columns={'final_account_id': 'Account.1_final_id'}, inplace=True)
data_merged_accounts.rename(columns={'final_bank_id': 'To Bank'}, inplace=True)
data_merged_accounts.drop(['account_id', 'bank_id'], axis=1, inplace=True)

## Save all accounts
accounts_df.drop(['bank_id', 'bank_id_int'], axis=1, inplace=True)
accounts_df.rename(columns={'final_account_id': 'acct_id'}, inplace=True)
accounts_df.rename(columns={'final_bank_id': 'bank_id'}, inplace=True)
accounts_df.to_csv(os.path.join(dir, "accounts.csv"), index=False)

## Save transactions
transactions_df = data_merged_accounts[['Account_final_id', 'Account.1_final_id']].copy()
transactions_df.columns = ['orig_acct', 'bene_acct'] 
transactions_df.drop_duplicates(inplace=True)  # eliminate repeat transactions
transactions_df.to_csv(os.path.join(dir, "transactions.csv"), index=False)


## Save laundering accounts
laundering_accounts = data_merged_accounts[data_merged_accounts['Is Laundering'] == 1][['Account_final_id', 'Account.1_final_id']]
laundering_accounts_df = pd.concat([laundering_accounts['Account_final_id'], laundering_accounts['Account.1_final_id']], ignore_index=True).drop_duplicates()
laundering_accounts_df.rename('acct_id', inplace=True)
laundering_accounts_df.to_csv(os.path.join(dir, "alert_accounts.csv"), index=False)