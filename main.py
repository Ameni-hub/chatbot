# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
url = 'https://www.gutenberg.org/files/6737/6737.txt'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text()
words = re.findall('\w' , text)
lowered = []
for word in words:
    lowered.append(word.lower())
text = nltk.Text(lowered)
fdist = nltk.FreqDist(text)
#fdist.plot(30)
#corpus = text
text_string = ' '.join(text)
sentences = sent_tokenize(text_string)
from contractions import contractions_dict


def preprocess(sentence):
    # Expand contractions
    words = word_tokenize(sentence)
    expanded_words = []
    for word in words:
        if word in contractions_dict:
            expanded_words.extend(contractions_dict[word].split())
        else:
            expanded_words.append(word)
    # Remove stopwords, punctuation, numbers, and lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in expanded_words if word.lower() not in stopwords.words('english') and word not in string.punctuation and not bool(re.match(r'\d+', word))]
    return words



# Define a function to find the most relevant sentence given a query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)

    # Preprocess each sentence in the text
    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]

    # Vectorize the corpus using TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    vectorized_corpus = vectorizer.fit_transform(preprocessed_sentences)

    # Vectorize the query
    query_vector = vectorizer.transform([query])

    # Compute the cosine similarity between the query and each sentence in the text
    similarities = cosine_similarity(query_vector, vectorized_corpus)[0]

    # Find the most similar sentence
    most_similar_index = similarities.argmax()
    most_relevant_sentence = sentences[most_similar_index]

    return most_relevant_sentence



def chatbot(question):
    #if isinstance(question, list):
        #queston = ' '.join(question)i
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence

# Create a Streamlit app


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    st.title(" Project Gutenberg ")
    st.write("Hello! I'm a chatbot. Ask me anything about :'The Social Cancer: A Complete English Version of Noli Me Tangere by José Rizal")
    # Get the user's question
    question = st.text_input("You:")
    # Create a button to submit the question
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot: " + response)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
