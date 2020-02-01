# -*- coding: utf-8 -*-
#######################ingesting data##########################################

import pandas as pd

apriori_df = pd.read_csv(r'apriori_run.csv',encoding='utf-8')

########################preprocessing for y##################################################

import re

y = apriori_df['fos']
y = y.apply(lambda x: re.sub('[^A-Za-z, +]', '', x))
y = y.apply(lambda x: re.findall(r'name([\w ]+),w',x))
data = y.tolist()

########################generate sparse df######################################

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(data).transform(data, sparse=True)
df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)

########################fpgrowth ###############################################

from mlxtend.frequent_patterns import fpgrowth

freq_items = fpgrowth(df, min_support=0.0001, use_colnames=True, max_len = 3, verbose = 3)
freq_items.to_csv('fpg_data.csv',index=False)

#########################assoc rules############################################

from mlxtend.frequent_patterns import association_rules

rules = association_rules(freq_items, metric ="lift", min_threshold = 0.9)
rules.to_csv('rules.csv',index=False)

########################pickle#################################################

import pickle

rule_df = open('rule_df','wb')
pickle.dump(rules,rule_df)
rule_df.close()
