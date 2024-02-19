import os
import re
from tqdm import tqdm
from pdfminer.high_level import extract_text
import spacy
from nltk.corpus import stopwords
import string
from string import digits
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from collections import defaultdict
from unidecode import unidecode
import enchant


class TextPreprocessing:        
    def __init__(self, path, spacyNLP=None):
        self.path = path
        self.nlp = spacyNLP if spacyNLP is not None else spacy.load('en_core_web_sm')  
        self.stop_words_list = stopwords.words("english")
        self.porter_stemmer = PorterStemmer()
        self.snowball_stemmer = SnowballStemmer('english')
        self.lancaster_stemmer = LancasterStemmer()
      
    def extract_text_from_pdfs(self):
        documents = {}
        if self.path:
            for anno in tqdm(os.listdir(self.path)):
                anno_path = os.path.join(self.path, anno)
                if os.path.isdir(anno_path):
                    for filename in os.listdir(anno_path):
                        if filename.endswith('.pdf'):
                            full_path = os.path.join(anno_path, filename)
                            text = extract_text(full_path)
                            if anno in documents:
                                documents[anno].append(text)  
                            else:
                                documents[anno] = [text]
        else:
            raise ValueError("No path specified for PDF extraction.")
        return documents
    
    def restructure_words(self, documents, wordlist, wordtoreplace):
        for year, papers in documents.items():
            for i, text in enumerate(papers):
                for word in wordlist:
                    if word in text:
                        text = text.replace(word, wordtoreplace)
                        documents[year][i] = text
        return documents

    def extract_core(self, documents, firstword, lastword):
        for year, papers in documents.items():
            for i, text in enumerate(papers):
                firstword_index = text.find(firstword)
                lastword_index = text.find(lastword) + len(lastword) 
                if firstword_index != -1 and lastword_index != -1 and lastword_index > firstword_index:
                    documents[year][i] = text[firstword_index+len(firstword):lastword_index-len(lastword)]
                elif firstword_index != -1:
                    documents[year][i] = text[firstword_index+len(firstword):]
                elif lastword_index != -1:
                    documents[year][i] = text[:lastword_index-len(lastword)]
        return documents

    def remove_end_from_txt(self, documents, wordlimit):
        for year, papers in documents.items():
            for i, text in enumerate(papers):
                if wordlimit in text:
                    idx = text.index(wordlimit)
                    if idx > len(text) * 0.8:                    
                           documents[year][i] = text[:idx]
        return documents

    def remove_empty_text(self, documents):
        new_docs = {}
        for year, papers in documents.items():
            for txt in papers:  
                if len(txt) > 0: 
                    if year in new_docs:
                        new_docs[year].append(txt)  
                    else:
                        new_docs[year] = [txt]
        return new_docs
    
    def unite_segmented_words(self, documents):
        new_paperz = {}
        for year, papers in documents.items():
            for txt in papers:
                new_paper = re.sub(r'-\n', '', txt)
                new_paperz.setdefault(year, []).append(new_paper)

        return new_paperz
    
    def comprehensive_cleaning(self, documents, replace_list, patterns):
        punctuation = string.punctuation
        new_docs = {}
        for year, papers in tqdm(documents.items()):
            for text in papers:
                text = text.replace(' ce ', ' circular economy ').replace(' CE ', 'circular economy')
                for element in replace_list:
                    text = text.replace(element, '')
                text = re.sub(r'http\S+|www\.\S+', '', text)
                text = text.translate(str.maketrans('', '', punctuation))
                text = text.translate(str.maketrans('', '', digits))
                for pattern in patterns:
                    text = re.sub(pattern, " ", text)
                text = unidecode(text)
                text = re.sub(r'\s+', ' ', text).lower().strip()
                new_docs.setdefault(year, []).append(text)
        return new_docs

    def tokenize_and_lemmatize(self, documents, base_form, stop_words_list=None, pos_filter=None):
        if stop_words_list is None:
            stop_words_list = self.stop_words_list
        tokenized_documents = {}
        for year, papers in tqdm(documents.items()):
                processed_papers = [] 
                for txt in papers:
                    tokens = []
                    doc = self.nlp(txt)  
                    for token in doc:
                        tk = token.text.lower()
                        if tk not in stop_words_list:
                            if base_form == "lemma":
                                tk = token.lemma_
                            elif base_form == "porter":
                                tk = self.porter_stemmer.stem(tk)
                            elif base_form == "snowball":
                                tk = self.snowball_stemmer.stem(tk)
                            elif base_form == "lancaster":
                                tk = self.lancaster_stemmer.stem(tk)
                            if pos_filter is None or token.pos_ in pos_filter:
                                tokens.append(tk)
                    processed_papers.append(tokens) 
                tokenized_documents[year] = processed_papers  
        return tokenized_documents
    
    
class TokenProcessor:
    def __init__(self, token_text, mesi, dictionary = None):
        self.token_text = token_text
        self.mesi = mesi
        self.dictionary = dictionary if dictionary is not None else enchant.Dict("en_US")
        self.nuovo_token_text = {}

    def check_and_split(self, word):
        words_found = []
        start_index = 0
        while start_index < len(word):
            found = False
            if (len(word) - start_index) < 3:
                break
            for end_index in range(len(word), max(2, start_index), -1):
                if self.dictionary.check(word[start_index:end_index]):
                    words_found.append(word[start_index:end_index])
                    start_index = end_index 
                    found = True
                    break
            if not found:
                start_index += 1
        return words_found

    def process_tokens(self):
        for year, liste in tqdm(self.token_text.items()):
            words_per_year = []
            for list_of_words in liste:
                words = []
                j = 0
                while j < len(list_of_words):
                    if list_of_words[j] in self.mesi:
                        words.append(list_of_words[j])
                        j += 1
                        continue

                    if j < len(list_of_words) - 1:
                        parola_unita = list_of_words[j] + list_of_words[j + 1]
                        if self.dictionary.check(parola_unita.lower()):
                            words.append(parola_unita)
                            j += 2
                            continue

                    if self.dictionary.check(list_of_words[j]):
                        words.append(list_of_words[j])
                        j += 1
                    else:
                        words += self.check_and_split(list_of_words[j])
                        j += 1

                words_per_year.append(words)
            self.nuovo_token_text[year] = words_per_year
