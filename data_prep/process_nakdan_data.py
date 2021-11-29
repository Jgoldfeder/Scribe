import sys,os,re,pathlib,torch
sys.path
sys.path.append('../')
sys.path.append('./')

MAX_SENTENCE_DISCARD_LENGTH = 61
MIN_SENTENCE_DISCARD_LENGTH = 2
MAX_SEQ_LEN = 512    

import data_prep.load_vocab as v
import data_prep.util as util

# POS
# bits 17-21
BASEFORM_POS = 0x1f0000
BASEFORM_POS_CONTINUE = 0x0
BASEFORM_POS_ADJECTIVE = 0x10000
BASEFORM_POS_ADVERB = 0x20000
BASEFORM_POS_CONJUNCTION = 0x30000
BASEFORM_POS_AT_PREP = 0x40000
BASEFORM_POS_NEGATION = 0x50000
BASEFORM_POS_NOUN = 0x60000
BASEFORM_POS_NUMERAL = 0x70000
BASEFORM_POS_PREPOSITION = 0x80000
BASEFORM_POS_PRONOUN = 0x90000
BASEFORM_POS_PROPERNAME = 0xa0000
BASEFORM_POS_CITATION = 0xb0000
BASEFORM_POS_INITIALISM = 0xc0000
BASEFORM_POS_VERB = 0xd0000
BASEFORM_POS_PUNCTUATION = 0xe0000
BASEFORM_POS_INTERROGATIVE = 0xf0000
BASEFORM_POS_INTERJECTION = 0x100000
BASEFORM_POS_UNKNOWN = 0x110000
BASEFORM_POS_QUANTIFIER = 0x120000
BASEFORM_POS_EXISTENTIAL = 0x130000
BASEFORM_POS_MODAL = 0x140000
BASEFORM_POS_PREFIX = 0x150000
BASEFORM_POS_URL = 0x160000
BASEFORM_POS_FOREIGN = 0x170000
BASEFORM_POS_JUNK = 0x180000
BASEFORM_POS_UNCLEAR = 0x190000
BASEFORM_POS_PARTICIPLE = 0x1a0000
BASEFORM_POS_COPULA = 0x1b0000
BASEFORM_POS_REF = 0x1c0000
BASEFORM_POS_TITULAR = 0x1d0000
BASEFORM_POS_SHEL_PREP = 0x1e0000
BASEFORM_POS_NONSTANDARD = 0x1f0000
pos_names = [ "CONTINUE", "ADJECTIVE", "ADVERB", "CONJUNCTION", "AT_PREP", "NEGATION", "NOUN", "NUMERAL", 
    "PREPOSITION", "PRONOUN", "PROPERNAME", "CITATION", "INITIALISM", "VERB", "PUNCTUATION", "INTERROGATIVE", 
    "INTERJECTION", "UNKNOWN", "QUANTIFIER", "EXISTENTIAL", "MODAL", "PREFIX", "URL", "FOREIGN", "JUNK", 
    "PARTICIPLE", "COPULA", "REF", "TITULAR", "SHEL_PREP", "UNCLEAR", "NONSTANDARD" ]

# Gender
#bits 22-23
BASEFORM_GENDER = 0x0000000000600000
BASEFORM_GENDER_MASCULINE = 0x0000000000200000
BASEFORM_GENDER_FEMININE = 0x0000000000400000
BASEFORM_GENDER_MASCULINEFEMININE = 0x0000000000600000
gender_names = [ "NONE", "MASCULINE", "FEMININE", "MASCULINEFEMININE"]

# Number
# bits 25-27
BASEFORM_NUMBER = 0x0000000007000000
BASEFORM_NUMBER_SINGULAR = 0x0000000001000000
BASEFORM_NUMBER_PLURAL = 0x0000000002000000
BASEFORM_NUMBER_DUAL = 0x0000000003000000
BASEFORM_NUMBER_DUALPLURAL = 0x0000000004000000
BASEFORM_NUMBER_SINGULARPLURAL = 0x0000000005000000
number_names = [ "", "SINGULAR", "PLURAL", "DUAL", "DUALPLURAL", "SINGULARPLURAL"]

# Person
# bits 28-30
BASEFORM_PERSON = 0x0000000038000000
BASEFORM_PERSON_1 = 0x0000000008000000
BASEFORM_PERSON_2 = 0x0000000010000000
BASEFORM_PERSON_3 = 0x0000000018000000
BASEFORM_PERSON_ANY = 0x0000000020000000
person_names = ["", "SINGULAR", "PLURAL", "DUAL", "DUALPLURAL", "SINGULARPLURAL" ]

# Status
# bits 31-32
BASEFORM_STATUS = 0x00000000c0000000
BASEFORM_STATUS_ABSOLUTE = 0x0000000040000000
BASEFORM_STATUS_CONSTRUCT = 0x0000000080000000
BASEFORM_STATUS_ABSOLUTECONSTRUCT = 0x00000000c0000000
status_names = [ "", "ABSOLUTE", "CONSTRUCT", "ABSOLUTECONSTRUCT"]

# Tense
# bits 34-36
BASEFORM_TENSE = 0x0000000e00000000
BASEFORM_TENSE_PAST = 0x0000000200000000
BASEFORM_TENSE_ALLTIME = 0x0000000400000000
BASEFORM_TENSE_PRESENT = 0x0000000600000000
BASEFORM_TENSE_FUTURE = 0x0000000800000000
BASEFORM_TENSE_IMPERATIVE = 0x0000000a00000000
BASEFORM_TENSE_TOINFINITIVE = 0x0000000c00000000
BASEFORM_TENSE_BAREINFINITIVE = 0x0000000e00000000
tense_names = ["", "PAST", "ALLTIME", "PRESENT", "FUTURE", "IMPERATIVE", "TOINFINITIVE", "BAREINFINITIVE" ]

# Prefix
# Function
# bits 2-3, 5,7, 9-11
PREFIX_FUNCTION_CONJUNCTION = 0x0000000000000002
PREFIX_FUNCTION_DEFINITEARTICLE = 0x0000000000000004
PREFIX_FUNCTION_INTERROGATIVE = 0x0000000000000010
PREFIX_FUNCTION_PREPOSITION = 0x0000000000000040
PREFIX_FUNCTION_RELATIVIZER_SUBORDINATINGCONJUNCTION = 0x00000000000000100
PREFIX_FUNCTION_TEMPORALSUBCONJ = 0x00000000000000200
PREFIX_FUNCTION_ADVERB = 0x00000000000000400
PREFIX_MASK = 0x0000000000000756
prefix_names = ["NOPREFIX", "CONJUNCTION", "DEFINITEARTICLE", "INTERROGATIVE", "PREPOSITION", "RELATIVIZER_SUB-CONJ", 
    "TEMPORALSUBCONJ", "ADVERB" ]
prefix_map = {}
prefix_map[0] = 0
prefix_map[PREFIX_FUNCTION_CONJUNCTION] = 1
prefix_map[PREFIX_FUNCTION_DEFINITEARTICLE] = 2
prefix_map[PREFIX_FUNCTION_INTERROGATIVE] = 3
prefix_map[PREFIX_FUNCTION_PREPOSITION] = 4

prefix_map[PREFIX_FUNCTION_RELATIVIZER_SUBORDINATINGCONJUNCTION] = 5
prefix_map[PREFIX_FUNCTION_TEMPORALSUBCONJ] = 6
prefix_map[PREFIX_FUNCTION_ADVERB] = 7

# Suffix: (gender,number,status,function), bits 37-38, 40-41, 43-48
SUFFIX_MASK = 0x0000fdb000000000

# Suffix Function
# bits 37-38
SUFFIX_FUNCTION = 0x0000003000000000
SUFFIX_FUNCTION_POSSESIVEPRONOUN = 0x0000001000000000
SUFFIX_FUNCTION_ACCUSATIVENOMINATIVE = 0x0000002000000000
SUFFIX_FUNCTION_PRONOMIAL = 0x0000003000000000
suffix_function_names = [ "", "POSSESIVEPRONOUN", "ACCUSATIVENOMINATIVE", "PRONOMINAL" ]

# Suffix Gender
# bits 40-41
SUFFIX_GENDER = 0x0000018000000000
SUFFIX_GENDER_MASCULINE = 0x0000008000000000
SUFFIX_GENDER_FEMININE = 0x0000010000000000
SUFFIX_GENDER_MASCULINEFEMININE = 0x0000018000000000
suffix_gender_names = ["", "SUF_GENDER_MASCULINE", "SUF_GENDER_FEMININE", "SUF_GENDER_MASCULINEFEMININE" ]

# Suffix Number
# bits 43-45
SUFFIX_NUMBER = 0x00001c0000000000
SUFFIX_NUMBER_SINGULAR = 0x0000040000000000
SUFFIX_NUMBER_PLURAL = 0x0000080000000000
SUFFIX_NUMBER_DUAL = 0x00000c0000000000
SUFFIX_NUMBER_DUALPLURAL = 0x0000100000000000
SUFFIX_NUMBER_SINGULARPLURAL = 0x0000140000000000
suffix_number_names = ["", "SUF_NUMBER_SINGULAR", "SUF_NUMBER_PLURAL", "SUF_NUMBER_DUAL", "SUF_NUMBER_DUALPLURAL", 
    "SUF_NUMBER_SINGULARPLURAL" ]

# Suffix Person
# bits 46-48
SUFFIX_PERSON = 0x0000e00000000000
SUFFIX_PERSON_1 = 0x0000200000000000
SUFFIX_PERSON_2 = 0x0000400000000000
SUFFIX_PERSON_3 = 0x0000600000000000
SUFFIX_PERSON_ANY = 0x0000800000000000
suffix_person_names = ["", "SUF_PERSON_1", "SUF_PERSON_2", "SUF_PERSON_3", "SUF_PERSON_ANY" ]

# Is the word Aramaic?
# bit 42
FULLFORM_ARAMAIC = 0x20000000000
aramaic_names = ["not aramaic", "aramaic"]

# Binyanim
# bits 52-54
BASEFORM_BINYAN = 0x0038000000000000
BASEFORM_BINYAN_PAAL = 0x0008000000000000
BASEFORM_BINYAN_NIFAL = 0x0010000000000000
BASEFORM_BINYAN_HIFIL = 0x0018000000000000
BASEFORM_BINYAN_HUFAL = 0x0020000000000000
BASEFORM_BINYAN_PIEL = 0x0028000000000000
BASEFORM_BINYAN_PUAL = 0x0030000000000000
BASEFORM_BINYAN_HITPAEL = 0x0038000000000000
binyan_names = ["", "BINYAN_PAAL", "BINYAN_NIFAL", "BINYAN_HIFIL", "BINYAN_HUFAL", "BINYAN_PIEL", "BINYAN_PUAL", "BINYAN_HITPAEL" ]

def decode_mask_print(mask):
    gender = (mask& BASEFORM_GENDER) >>21 
    print(gender_names[gender])
    pos = (mask& BASEFORM_POS) >>16 
    print(pos_names[pos])
    number = (mask& BASEFORM_NUMBER) >>24
    print(number_names[number])        
    person = (mask& BASEFORM_PERSON) >>27
    print(person_names[person]) 
    status = (mask& BASEFORM_STATUS) >>30
    print(status_names[status])     
    tense = (mask& BASEFORM_TENSE) >>33
    print(tense_names[tense])
    prefix = prefix_map[(1 << (mask&PREFIX_MASK).bit_length())//2]
    print(prefix_names[prefix])
    suffix_function = (mask& SUFFIX_FUNCTION) >>36 
    print(suffix_function_names[suffix_function])
    suffix_gender = (mask& SUFFIX_GENDER) >>39 
    print(suffix_gender_names[suffix_gender])
    suffix_number = (mask& SUFFIX_NUMBER) >>42
    print(suffix_number_names[suffix_number])
    suffix_person = (mask& SUFFIX_PERSON) >>45
    print(suffix_person_names[suffix_person])
    is_aramaic = 1 if mask & FULLFORM_ARAMAIC else 0
    print(aramaic_names[is_aramaic])
    binyan = (mask& BASEFORM_BINYAN) >>51
    print(binyan_names[binyan])

def decode_mask(mask):
    gender = (mask& BASEFORM_GENDER) >>21 
    pos = (mask& BASEFORM_POS) >>16 
    number = (mask& BASEFORM_NUMBER) >>24
    person = (mask& BASEFORM_PERSON) >>27
    status = (mask& BASEFORM_STATUS) >>30
    tense = (mask& BASEFORM_TENSE) >>33
    prefix = prefix_map[(1 << (mask&PREFIX_MASK).bit_length())//2]
    suffix_function = (mask& SUFFIX_FUNCTION) >>36 
    suffix_gender = (mask& SUFFIX_GENDER) >>39 
    suffix_number = (mask& SUFFIX_NUMBER) >>42
    suffix_person = (mask& SUFFIX_PERSON) >>45
    is_aramaic = 1 if mask & FULLFORM_ARAMAIC else 0
    binyan = (mask& BASEFORM_BINYAN) >>51
    return [gender, pos, number, person, status, tense, prefix, suffix_function, suffix_gender, suffix_number, suffix_person, is_aramaic, binyan]

class Vocabulary:
    def __init__(self):    
        embeddings, vocab, word_list = v.load_vocab()
        self.vocab = vocab
    def get_word_index(self,word):
        # remove nikud
        word = re.sub("[\u0591-\u05c7]", "", word)      
        # remove underscores (nakdan adds them before commas, unsure why )
        word = word.replace("_","")
        if word == "," or word == ":":
            index = self.vocab["<COMMA>"]
        elif word == ".":
            index = self.vocab["<STOP>"]
        elif word == "?":
            index = self.vocab["<QUES>"]
        # prefix split with |
        fullword = word.replace("|","") 
        if fullword in self.vocab:
            index = self.vocab[fullword]
        elif "|" in word and word.split("|")[1] in self.vocab:
            index = self.vocab[word.split("|")[1]]
        else:
            index = self.vocab["<UNK>"]
        return index
    
def process_sentence(sentence, vocab,reset_at_phrases):
    pos = 0
    tags = []
    words = []
    if len(sentence) <= MIN_SENTENCE_DISCARD_LENGTH or len(sentence) > MAX_SENTENCE_DISCARD_LENGTH or "." in sentence[0] or "," in sentence[0]:
        return [],[]
    for line in sentence:
        if line.strip()=="":
            continue
        tokens = line.split()
        word = tokens[0]
        
        mask = int(tokens[1])

        index = vocab.get_word_index(word)
        word_data = decode_mask(mask) + [index] + [pos]

        if not "." in word and not "," in word:
            pos += 1
            tags.append(0)
            words.append(torch.tensor(word_data))

        if "," in word:
            if reset_at_phrases:
                pos = 0
            tags[-1] = 1
        elif "." in word:
            tags[-1] = 2   
            if reset_at_phrases:
                pos = 0            
        
    # last word always ends sentence
    tags[-1] = 2   
    return words, tags



def process_file(file_name, vocab, reset_at_phrases):
    with open(file_name, 'r', encoding='utf-8')  as file:       
        sentence = []
        words = []
        tags = []
        for line in file:
            # every new sentence has a line starting with >>
            # sentences to be ignored will contain a #
            if line.startswith(">>"):
                if (len(sentence) > 0) and not "#" in sentence[0]:
                    words_, tags_ = process_sentence(sentence,vocab, reset_at_phrases)
                    words += words_
                    tags += tags_
                        
                sentence = []
            else:
                sentence.append(line)
    return words, tags   

def y_chunks(lst, n):
    return [torch.tensor(lst[i:i + n]) for i in range(0, len(lst), n)]

def x_chunks(lst, n):
    return [torch.stack(lst[i:i + n]) for i in range(0, len(lst), n)]

def split_lists(x,y,seq_len):
    if len(x) != len(y):
        raise ValueError((len(x),len(y)))
    extra = len(x) % seq_len
    extra_x = []
    extra_y = []
    if extra!=0:        
        extra_x = x[-extra:]
        x = x[:-extra]
        extra_y = y[-extra:]
        y = y[:-extra]
        
    return x_chunks(x,seq_len), extra_x, y_chunks(y,seq_len), extra_y
    

def process_dir(src,seq_len, reset_at_phrases):
    path = "nakdan_preprocessed_data"+os.sep+ src 
    vocab = Vocabulary()
    X = []
    Y = []
    words = []
    tags = []
    for file in pathlib.Path(path).glob("*.txt"):
        print(file)
        words_, tags_ = process_file(file,vocab,reset_at_phrases)
        words+=words_
        tags+=tags_
        x, ex,y,ey  = split_lists(words,tags,seq_len)      
        X += x
        Y += y
        words= ex
        tags = ey
    X = torch.stack(X)
    Y = torch.stack(Y)
    return X,Y

def process_dirs(seq_len,name, reset_at_phrases):
    X = []
    Y = []
    for directory in ["bar_ilan","halacha","liturgy","mussar","philosophy","responsa","talmud","tanach_commentary"]:
        x,y =process_dir(directory,seq_len, reset_at_phrases)
        X.append(x)
        Y.append(y)
        print(x.shape,y.shape)
    X = torch.cat(X)
    Y = torch.cat(Y)
    print(X.shape,Y.shape)
    if reset_at_phrases:
        util.save_tensor(X,str(seq_len)+name+"_x_phrase.pt")
        util.save_tensor(Y,str(seq_len)+name+"_y_phrase.pt")    
    else:
        util.save_tensor(X,str(seq_len)+name+"_x_sentence.pt")
        util.save_tensor(Y,str(seq_len)+name+"_y_sentence.pt")
    return X,Y   
#process_dirs(MAX_SEQ_LEN, "test_dataset")    

def get_sentence_dataset(seq_len,name):
    try:
        return util.load_tensor(str(seq_len)+name+"_x_sentence.pt"), util.load_tensor(str(seq_len)+name+"_y_sentence.pt")
    except:
        return process_dirs(seq_len, name, False)  

def get_phrase_dataset(seq_len,name):
    try:
        return util.load_tensor(str(seq_len)+name+"_x_phrase.pt"), util.load_tensor(str(seq_len)+name+"_y_phrase.pt")
    except:
        return process_dirs(seq_len, name,True) 