# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:11:45 2019

@author: Nikhil Vishwanath, 
         M.Sc. Data Science,        
         Chennai Mathematical Institute
"""
print("Give input in following format")
print("k_itemset(k, frequency, vocab file address, docword file address)")

def k_itemset(k, f, vocab, words):
    import time
    import pandas as pd
    
    #to start the timer 
    start = time.time()
    
    #reading the vocab file 
    r = pd.read_csv(vocab,sep = " ",header = None) 
    r = pd.DataFrame(r)
    #increasing the index by one so that index is same as wordID 
    #which will save storage space 
    
    r.index = range(1,len(r)+1)
    
    #reading docword file 
    doc = pd.read_csv(words)
    
        
    df =doc.loc[2:,:] 
    
    #reanming column, because while reading the file it read first entry as column name
    df = df.rename(columns = {df.columns[0]:"A"})
    
    #since the file was read as one column dataset, all the lines from 2 onwards have
    #three space seperated integers respectively docID, wordID, count 
    #the line below will make three columns in dataframe 
    #split each row into three parts and then store values in respective columns
    
    doc = pd.DataFrame(df.A.str.split(" ",2).tolist(),columns = ["docID","wordID","count"])
    doc = pd.DataFrame(doc)
    
    #then compiling all the words that occur in given document into a single tuple 
    #and storing the values as list of tuples where each tuple consist of all the words in 
    #respective document. These tuples are our transactions.
    
    
    
    #This for lop was not efficient so I had to drop it
    '''
    for i in doc["docID"].unique():
        d.append(tuple(set( [doc["wordID"][j] for j in doc[doc["docID"]==i].index])))
    '''
    
    #Storing in list of tuples

    d = doc.groupby("docID")["wordID"].apply(tuple)
    
    
    #importing the package efficient_apriory and running apriori algorithm on our transactions
    from efficient_apriori import itemsets_from_transactions as item
    #item function takes three inputs (list of transactions , min support or frequency, max k frequent itemset)
    itemsets = item(d, f, k+1)
    
    #Earlier I had mapped wordIDs to words while making list of transactions but it was
    #very costly in terms of compution and memory, therefore I changed the code such that 
    #each transaction contains wordIDs and when we get the answer I mapped wordIds to corresponding words
    
    
   
    # calculating the end time 
    end = time.time()
    
    #if k-itemset is not present for the given freq, it will terminate     
    
    if len(itemsets[0])<k:
        print("Time taken is", end-start,"sec")
        return "No K-itemset for given freq"
    #if k-itemset is present it will return all the k-itemsets and the time for execution
    else:
        
        #mapping wordIDs to corresponding words
        ans = [[r.loc[int(list(itemsets[0][k].keys())[j][i])][0] for i in range(k)] for j in range(len(list(itemsets[0][k].keys())))]
    
        print("Time taken is", end-start,"sec")
        return ans 
    