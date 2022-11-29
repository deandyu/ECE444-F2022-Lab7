from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

from flask import Flask, request

##### model loading #####
loaded_model = None
with open('basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

vectorizer = None
with open('count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)

application = Flask(__name__)

@application.route("/api", methods=["GET"])
def fact_check():
    text = request.args.get('text')
    prediction = loaded_model.predict(vectorizer.transform([text]))[0]

    # Convert output label to 1 for fake news, 0 otherwise
    # As specified in the lab assignment
    if(prediction == "FAKE"):
        label = 1
    elif(prediction == "REAL"):
        label = 0

    return str(label)

if __name__ == "__main__":
    application.run()