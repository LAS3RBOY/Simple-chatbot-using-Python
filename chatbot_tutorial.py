import numpy as np
import nltk
import string
import random


#importing and reading the corpus
f=open('chatbot.txt','r',errors='ignore')
raw_doc=f.read()
raw_doc=raw_doc.lower()
#nltk.download('punkt')
#nltk.download('wordnet')
sent_token=nltk.sent_tokenize(raw_doc)
word_token=nltk.word_tokenize(raw_doc)


sent_token[:2]
print(sent_token)


word_token[:2]
print(word_token)


# text processing
lemmer= nltk.stem.WordNetLemmatizer()
# wordnet is a semantically oriented dictionary of english included in the nltk
def lemTokens(tokens):
    return [lemmer.lemmatize(token)for token in tokens]
remove_punch_dict = dict((ord(punct),None)for punct in string.punctuation)
def lemNormalize(text):
    return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punch_dict)))


# defining the greeting function
greetInputs= ("hi",'hello',"hey","hii","How are you?","Hi",'Hey','Hello','sup',"what's up",'greeting')
greetResponse=('hii','hey','hii there','Hello','hii there','I am glad ! you are talking to me')
def greet(sentence):
    for words in sentence.split():
        if words.lower() in greetInputs:
            return random.choice(greetResponse)



#Response Generation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo1_response=' '
    TfidfVec= TfidfVectorizer(tokenizer=lemNormalize,stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_token)
    vals = cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat=vals.flatten()
    flat.sort()
    req_tfidf=flat[-2]
    if(req_tfidf == 0):
        robo1_response=robo1_response+"I am sorry! I dont get your words"
        return robo1_response
    else:
        robo1_response=robo1_response+sent_token[idx]
        return robo1_response


#Defining conversation start/end protocols

flag=True
print("BOT: My name is Jarvis. Let's have a conversation! Also, if you want to exit any time, just type bye ")
while (flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag =False
            print("BOT: You are Welcome")

        else:
            if(greet(user_response)!=None):
                print("BOT: "+greet(user_response))
            else:
                sent_token.append(user_response)
                word_token=word_token+nltk.word_tokenize(user_response)
                final_word=list(set(word_token))
                print("BOT: ",end=" ")
                print(response(user_response))
                sent_token.remove(user_response)

    else:
        flag=False
        print("BOT: , GoodBye,TakeCare <3")