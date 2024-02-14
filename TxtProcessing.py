
import pandas as pd
import os
import copy
import re
import string
from tqdm import tqdm
from unstructured.cleaners.extract import extract_ip_address
from unstructured.cleaners.core import clean_extra_whitespace
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text

class txtpreprocessing:
    def __init__(self, path):
        self.path = path
        self.papers = []
        self.core = [] 
        self.papers_no_end = []
        self.ncore = []
        self.new_txts = []
        self.cleaned_text = []  
        
    def extract_text(self, path): 
        for anno in tqdm(os.listdir(self.path)):
            anno_path = os.path.join(self.path, anno)
            if os.path.isdir(anno_path):
                for filename in os.listdir(anno_path):
                    if filename.endswith('.pdf'):
                        full_path = os.path.join(anno_path, filename)
                        text = extract_text(full_path)
                        paper = [text]  
                        self.papers.append(paper)
        
        return self.papers
    
    
    def restructure_words(self, wordlist, wordtoreplace):
        for i, paper in enumerate(self.papers):
            for j, text in enumerate(paper):
                for e, word in enumerate(wordlist):
                    if word in text:
                        self.papers[j][i] = text.replace(word, wordtoreplace)
        
        return self.papers
    
    def extract_core(self, firstword, lastword):
        for paper in self.papers:
            paper_content = [] 
            for text in paper:
                firstword_index = text.find(firstword)
                lastword_index = text.find(lastword)
                if firstword_index != -1 and lastword_index != -1:
                    content = text[firstword_index:lastword_index]
                    paper_content.append(content)  
                elif firstword_index != -1:
                    content = text[firstword_index:]
                    paper_content.append(content) 
                elif lastword_index != -1:
                    content = text[:lastword_index]
                    paper_content.append(content)  
                else:
                    paper_content.append(text)
            if paper_content:  
                core.append(paper_content)
        
        return self.core  
    
    def remove_end_from_txt(self, wordlimit):
        for i, p in enumerate(self.core):
            for j, x in enumerate(p):
                new_txt = x 
                if wordlimit in x:
                    idx = x.index(wordlimit)
                    if idx > len(x) * 0.8:
                        new_txt = x[:idx]
                self.papers_no_end.append([new_txt])

        return self.papers_no_end
                
    def remove_empty_text(self):
        delete = []
        for i, paper in enumerate(self.core):
            for text in paper:
                if len(text) == 0:
                    print(i)
                    delete.append(text)
        
        self.ncore = [[txt] for paper in self.core for txt in paper if txt not in delete]
        
        return self.ncore
        
    def remove_patterns(self, patterns):
        for paper in ncore: 
            for txt in paper:
                new_txt = txt 
                for pattern in patterns:  
                    new_txt = re.sub(pattern, " ", new_txt)
                new_txt = re.sub(r'\s+', ' ', new_txt).strip()
                new_txts.append([new_txt])
        
        return self.new_txts
   

    def comprehensive_cleaning(self):
        # replace formatting characters 
        cleaned_text = [[text.replace('\n', ' ').replace('\t', ' ').replace('\x0c', '')] for paper in self.new_txts for text in paper]
        # clean non ascii characters
        cleaned_text = [[clean_non_ascii_chars(text)] for paper in cleaned_text for text in paper]
        # clean unicode text
        cleaned_text = [[replace_unicode_quotes(text)] for paper in cleaned_text for text in paper]
        # remove ip adress
        ip_address = []
        for paper in cleaned_text:
            for text in paper:
                address = extract_ip_address(text)
                if len(address) > 1:
                    ip_address.append(address)
        
        cleaned_text = [[element] for paper in cleaned_text for element in paper if element not in ip_address]
        # remove urls
        cleaned_text = [[re.sub(r'http\S+|www\.\S+', '', txt)] for paper in cleaned_text for txt in paper]
        # remove extra-white-space
        cleaned_text = [[clean_extra_whitespace(text)] for paper in cleaned_text for text in paper]
        # remove punctuation
        table = str.maketrans('', '', string.punctuation)
        cleaned_text = [[text.translate(table)] for paper in cleaned_text for text in paper]
        # replace instances of ' ce ' with 'circular economy'
        cleaned_text = [[text.replace(' ce ', ' circular economy ')] for paper in cleaned_text for text in paper]
        # lower case the whole text
        cleaned_text = [[txt.lower()] for paper in cleaned_text for txt in paper]
        
        self.cleaned_text = cleaned_text
        return self.cleaned_text

