from flask import Flask, render_template, request
from pythainlp.tokenize import word_tokenize
import pickle

# โหลดโมเดลและ TF-IDF ที่ฝึกไว้
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Flask app
app = Flask(__name__)

# ฟังก์ชันตัดคำ
def clean_and_tokenize(text):
    tokens = word_tokenize(text, engine='newmm')
    return ' '.join(tokens)

# Route หน้าเว็บ
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['news']
        cleaned = clean_and_tokenize(user_input)
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        prediction = 'ข่าวปลอม (Fake News)' if pred == 1 else 'ข่าวจริง (Real News)'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# from flask import Flask, request, jsonify
# import joblib
# from pythainlp.tokenize import word_tokenize

# # โหลดโมเดลและ TF-IDF vectorizer
# model = joblib.load('fake_news_model.pkl')
# tfidf = joblib.load('tfidf_vectorizer.pkl')

# app = Flask(__name__)

# # ฟังก์ชันตัดคำ
# def clean_and_tokenize(text):
#     tokens = word_tokenize(text, engine='newmm')
#     return ' '.join(tokens)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if not data or 'text' not in data:
#         return jsonify({'error': 'Missing "text" field'}), 400
    
#     text = clean_and_tokenize(data['text'])
#     vec = tfidf.transform([text])
#     prediction = model.predict(vec)[0]
#     return jsonify({'fake_news': bool(prediction)})

# if __name__ == '__main__':
#     app.run(debug=True)
