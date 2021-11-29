import sys,re, string
sys.path
sys.path.append('../')
sys.path.append('./')

from flask import Flask, request, render_template, render_template_string
import data_prep.load_vocab as v
import torch
import models.model as models
import data_prep.process_nakdan_data as pnd

import nakdan_server.nakdan_server as tagger


app = Flask(__name__)
vocab = pnd.Vocabulary()
embeddings, _vocab, word_list = v.load_vocab()
phrase_model = models.BLSTM(2,embeddings)
phrase_model.load("phrase_tagger_9452.pt")
sentence_model = models.BLSTM(2,embeddings)
sentence_model.load("sentence_tagger_9775.pt")

def preprocess_line(line):
    # change different types of dashes to spaces 
    line = re.sub('[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]'," ",line)
    # replace gershayim and quote variants
    line = re.sub('[\u201c\u201d\u05F4]','"',line)
    # replace geresh and single quote variants
    line = re.sub('[\u2018\u2019\u0060\u05f3]',"'",line)
    
    output = []
    punctuation = ""
    words = line.split()
    for word in words:
        # filter out English words and numbers
        if re.search('[a-zA-Z0-9]', word):
            continue
        
        # remove vowels
        word = re.sub("[\u0591-\u05c7]", "", word)
        punctuation = ""
        if "." in word:
            punctuation = "."
        if "," in word:
            punctuation = ","
        if "?" in word:
            punctuation = "?"
        if ":" in word:
            punctuation = ":"
        if ";" in word:
            punctuation = ";"  
        if "!" in word:
            punctuation = "!"               
        # remove punc, but not double quotes or single quotes (equivalent of geresh and gersgayim)
        word = word.translate(str.maketrans('', '', "׳"+string.punctuation.replace("\"","").replace("'","")))  
        # remove double quotes at beginning and end of word
        word = word.strip('"')
        # remove LRM and RLM
        word = word.strip('\u200e').strip('\u200f')
        # remove all non hebrew or " or '
        word = re.sub('[^\u0027"\u0590-\u05fe]+', "", word)
        if word == "":
            continue
        
        output.append(word)
        if punctuation:
            output.append(punctuation)
    return output

def process_sentence(sentence_tokens, vocab):
    punc = ",.;?!:"
    sentence = ""
    for word in sentence_tokens:
        if any(p in word for p in punc):
            sentence += " " + word
        else:
            sentence += " " + word
    sentence.strip()
    tagged = tagger.tag(sentence)
    res = []
    for l in tagged:
        word = l[0]
        word_idx = vocab.get_word_index(word)
        res.append((word,word_idx,l[1]))
    return res




def preprocess_string(string, vocab):
    tags = {}
    word_pos = 0
    end_of_sentence_punc = "!.;?:"
    
    results = []
    lines = string.split("\n")
    sentence = []
    for line in lines:
        # if line contains a ###, skip it. This indicates a comment
        if '###' in line:
            continue

        words = preprocess_line(line)
        for word in words:
            if any(punc in word for punc in end_of_sentence_punc):
                # for simplicity, append period for all end of sentence tokens, except question marks
                if "?" in word:
                    sentence.append("?")
                else:
                    sentence.append(".")
                
                # get tags of existing punctuation
                tags[word_pos-1] = 1
                
                # process sentence
                sentence = process_sentence(sentence,vocab)
                # write new sentence to file
                results.append(sentence)
                # start a new sentence
                sentence = []
            elif "," in word:
                # get tags of existing punctuation
                tags[word_pos-1] = 2
            else:
                sentence.append(word)
                word_pos+=1

    if len(sentence) > 0:
        sentence = process_sentence(sentence,vocab)
        results.append(sentence)
    return results,tags    
    



def punctuate(text, add_periods, add_comas, keep_original_text, sentence_first):
    print("add_periods, add_comas, keep_original_text, sentence_first",add_periods, add_comas, keep_original_text, sentence_first)
    preprocessed_text,tags = preprocess_string(text,vocab)
    result = []
    for sentence in preprocessed_text:
        for line in sentence:
            word = line[0]
            if word.strip() == "":
                continue
   
            mask = line[2]
            vocab_index = line[1]
            word_data = pnd.decode_mask(mask) + [vocab_index] 

            result.append(torch.tensor(word_data))
            
    result = torch.stack(result)
    
    # get sentences
    if not keep_original_text:
        tags = {}
    words = text.split()

    if add_periods and add_comas:
        if sentence_first:
            just_period_tags = {}
            for key in tags.keys():
                if tags[key] == 1:
                    just_period_tags[key] = 1
            seq = sentence_model.decode(result,just_period_tags)
            for i in range(len(words)):
                if seq[i] == 0:
                    tags[i-1] = 1
            all_comma_tags = {}
            for key in tags.keys():
                if tags[key] == 1 or tags[key] == 2:
                    all_comma_tags[key] = 1
            seq = phrase_model.decode(result,all_comma_tags) 
            for i in range(len(words)):
                if seq[i] == 0:
                    if (i-1) not in tags:
                        tags[i-1] = 2            
        
        else:
            zero_tags = {}
            all_comma_tags = {}
            for key in tags.keys():
                if tags[key] == 1 or tags[key] == 2:
                    all_comma_tags[key] = 1
            seq = phrase_model.decode(result,all_comma_tags)        
            for i in range(len(words)):
                if seq[i] == 0:
                    if (i-1) not in tags:
                        tags[i-1] = 2  
                else:
                  zero_tags[i-1] = 0   
            seq = sentence_model.decode(result,zero_tags)
            for i in range(len(words)):
                if seq[i] == 0:
                    if (i-1) not in tags:
                        raise ValueError()
                    tags[i-1] = 1
    elif add_comas:
        all_comma_tags = {}
        for key in tags.keys():
            if tags[key] == 1 or tags[key] == 2:
                all_comma_tags[key] = 1
        seq = phrase_model.decode(result,all_comma_tags)        
        for i in range(len(words)):
            if seq[i] == 0:
                if (i-1) not in tags:
                    tags[i-1] = 2   
    elif add_periods:
        just_period_tags = {}
        for key in tags.keys():
            if tags[key] == 1:
                just_period_tags[key] = 1
        seq = sentence_model.decode(result,just_period_tags)
        for i in range(len(words)):
            if seq[i] == 0:
                tags[i-1] = 1
    result = ""
    for i in range(len(words)):
        word = words[i].replace(".","").replace(":","").replace(",","").replace("?","").replace("!","")
        result +=" "+word
        if i not in tags:
            continue
        if tags[i] == 1:
            result +="."
        if tags[i] == 2:
            result +=","
        
    #    if seq[i] == 0:
    #        tags[i-1] = 1
    #    if seq[i] == 0:
    #        result +="."
    #    word = words[i].replace(".","").replace(":","")
    #    result +=" "+word
    
    return result


@app.route("/")
def home():
    return render_template("home.html",original="",data="")
    
@app.route("/about")
def about():
    return render_template("about.html")
    
@app.route('/', methods=['POST'])
def home_post():
    text = request.form['text']
    add_periods = request.form.get('period') != None
    add_comas = request.form.get('coma') != None
    keep_original_text = request.form.get('original') != None
    sentence_first = request.form.get('sentence_first') != None

    result = punctuate(text, add_periods, add_comas, keep_original_text, sentence_first)
    return render_template("home.html",original=text,data=result)
    
def run():
    app.run(debug=True)
    
if __name__ == "__main__":
    app.run(debug=True)