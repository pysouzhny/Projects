import json
from pprint import pprint as pp
import pandas as pd
import os

os.chdir(r"C:\Users\Dominique Njinkeu\Documents")
count=100000*20
tweets = []
lst=0

def read_in_chunks(infile,chunk_size=1024*64):
    chunk=infile.read(chunk_size)
    while chunk:
        yield chunk
        chunk=infile.read(chunk_size)

with open("tweets.json") as f:
    
        
    for line in f:
        if count>0:
            if line.strip() == "": continue
            
            try:
                tweet=json.loads(line)
  
            except Exception as e:
                #print e
                lst+=1
                
                continue
            if tweet.get('id_str') == None: continue
            if tweet["created_at"].startswith("W"):continue
            if tweet["created_at"].startswith("T"):continue
            if tweet["created_at"].startswith("F"):continue
            if tweet["created_at"].startswith("Sa"):continue
            if tweet["created_at"]==None:continue
            #if tweet["coordinates"]==None or tweet["coordinates"]=="FALSE":continue
            tweets.append(tweet)
            count-=1

print "starting errors logging"
print "number of errors encountered "+str(lst)
import csv
print "starting opening file"
import random
k=random.sample(range(len(tweets)),3000)
Num=sorted(k)
line_counter=0
with open("tweets.csv","wb") as csvfile:
    f=csv.writer(csvfile)


    f.writerow([key for key in tweet.keys()])
   
    for tweet in tweets:
        line_counter+=1
        print line_counter
        if line_counter==Num[0]:
            f.writerow([unicode(tweet[key]).encode("utf-8") for key in tweet])
            Num.remove(Num[0])
            if len(Num)==0:
                break


        
        



        
        