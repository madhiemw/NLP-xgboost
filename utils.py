import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
tknzr = TweetTokenizer()
wnl = WordNetLemmatizer()
ps = PorterStemmer()

def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

def remove_punctuation(text):
    no_punct = [words for words in text if words not in string.punctuation]
    words_wo_punct = ''.join(no_punct)
    words_wo_br = words_wo_punct.replace('br', '').replace('br', '')
    return words_wo_br

def preprocess_text(text):
    text = remove_html_tags(text)
    text = remove_punctuation(text)
    text = text.lower()
    return text

def stem_tokens(tokens):
    if isinstance(tokens, list):
        tokens = ' '.join(tokens)
    stemmed_words = [ps.stem(word) for word in tokens.split()]
    return stemmed_words

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    cleaned_text = [word for word in tokens if word.lower() not in stop_words]
    return cleaned_text

def tokenize_text(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return tokens

def lemmatize_tokens(tokens):
    return [wnl.lemmatize(token) for token in tokens]

def untokenize(tokens):
    return ' '.join(tokens)
