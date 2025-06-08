from flask import Flask, render_template, request
from pythainlp.tokenize import word_tokenize
from langdetect import detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pickle
import torch
import os

# โหลดโมเดลภาษาอังกฤษ
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
translator_model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# โหลดโมเดล Fake News ที่ฝึกไว้
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

app = Flask(__name__)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # fallback

def translate_to_english(text, source_lang):
    tokenizer.src_lang = source_lang
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = translator_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id("en")
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def clean_and_tokenize(text):
    tokens = word_tokenize(text, engine='newmm')
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    translated_text = ""
    if request.method == 'POST':
        user_input = request.form['news']
        lang = detect_language(user_input)
        
        if lang != "en":
            translated_text = translate_to_english(user_input, lang)
        else:
            translated_text = user_input

        # ตัดคำภาษาไทย (ยังใช้สำหรับ input ไทย)
        if lang == "th":
            cleaned = clean_and_tokenize(user_input)
        else:
            cleaned = translated_text  # ภาษาอื่นไม่ต้องตัดคำด้วย pythainlp

        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        prediction = 'Fake News' if pred == 1 else 'Real News'

    return render_template('index.html', prediction=prediction, translated=translated_text)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# from flask import Flask, render_template, request
# from pythainlp.tokenize import word_tokenize
# import pickle

# # โหลดโมเดลและ TF-IDF ที่ฝึกไว้
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# with open('tfidf.pkl', 'rb') as f:
#     tfidf = pickle.load(f)

# # Flask app
# app = Flask(__name__)

# # ฟังก์ชันตัดคำ
# def clean_and_tokenize(text):
#     tokens = word_tokenize(text, engine='newmm')
#     return ' '.join(tokens)

# # Route หน้าเว็บ
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction = None
#     if request.method == 'POST':
#         user_input = request.form['news']
#         cleaned = clean_and_tokenize(user_input)
#         vec = tfidf.transform([cleaned])
#         pred = model.predict(vec)[0]
#         prediction = 'ข่าวปลอม (Fake News)' if pred == 1 else 'ข่าวจริง (Real News)'
#     return render_template('index.html', prediction=prediction)

# import os

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))  # Render จะกำหนด PORT มาเอง
#     app.run(host='0.0.0.0', port=port)


