import email
import nltk
import os
import pickle
import string

nltk.download('stopwords')
nltk.download('punkt')

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()


def read_labels(path):
    "Read the mail labels."
    labels = {}

    with open(path) as f:
        for line in f:
            label, key = line.strip().split()
            labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

    return labels


def flatten_to_string(parts):
    "Combine the different parts of the email into a flat list of strings."
    ret = []
    if type(parts) == str:
        ret.append(parts)

    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)

    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()

    return ret


def extract_email_text(path):
    "Extract subject and body text from a single email file."

    with open(path, errors='ignore') as f:
        msg = email.message_from_file(f)

    if not msg:
        return ""

    subject = msg['Subject'] if msg['Subject'] else ""

    body = ' '.join(
        m for m in flatten_to_string(msg.get_payload())
        if type(m) == str
    )

    if not body:
        body = ""

    return subject + ' ' + body


def load_stems(path):
    "Process a single email file into stemmed tokens"

    email_text = extract_email_text(path)

    if not email_text:
        return []

    tokens = nltk.word_tokenize(email_text)
    tokens = [i.strip("".join(punctuations))
              for i in tokens if i not in punctuations]

    if len(tokens) > 2:
        return set([stemmer.stem(w) for w in tokens if w not in stopwords])

    return []
