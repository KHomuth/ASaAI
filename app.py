## Code erstellt nach Tutorial, identsiche Codezeilen nach:
## Parashar, A. (2020, 13. Oktober). Sentiment Analysis web app using NLTK and Heroku. medium. https://medium.com/pythoneers/sentiment-analysis-web-app-using-nltk-and-heroku-96ccd37c44ef

from flask import Flask,render_template,request
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

###Loading model and cv
cv = pickle.load(open('sentiment_analysis/cv.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        review = request.form['review']
        new_review = re.sub('[^a-zA-Z]', ' ', review)
        new_review = new_review.lower()
        new_review = new_review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = cv.transform(new_corpus).toarray()

        if request.form['algorithm'] != 'all':
            new_model = request.form['algorithm']
            model = pickle.load(open('sentiment_analysis/' + new_model + '_review.pkl','rb'))
            pred = model.predict(new_X_test)
            return render_template('result.html',review=review,review_result=new_review,x_test=new_X_test,prediction=pred,name=new_model)
        else:
            model_nb = pickle.load(open('sentiment_analysis/nb_review.pkl','rb'))
            model_lr = pickle.load(open('sentiment_analysis/lr_review.pkl','rb'))
            model_dtc = pickle.load(open('sentiment_analysis/dtc_review.pkl','rb'))
            model_rfc = pickle.load(open('sentiment_analysis/rfc_review.pkl','rb'))
            pred_nb = model_nb.predict(new_X_test)
            pred_lr = model_lr.predict(new_X_test)
            pred_dtc = model_dtc.predict(new_X_test)
            pred_rfc = model_rfc.predict(new_X_test)
            return render_template('result_all.html',review=review,review_result=new_review,x_test=new_X_test,prediction_nb=pred_nb,prediction_lr=pred_lr,prediction_dtc=pred_dtc,prediction_rfc=pred_rfc)


if __name__ == "__main__":
    app.run(debug=True)