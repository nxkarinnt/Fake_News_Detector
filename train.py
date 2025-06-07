import pandas as pd
from pythainlp.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- โหลดข้อมูล ----------
df = pd.read_csv("data/fakeNews.csv", encoding="utf-8-sig")

# ---------- รวมคอลัมน์ข้อความ ----------
df['full_text'] = df[['title', 'text', 'subject', 'date']].fillna('').agg(' | '.join, axis=1)

# ---------- เตรียม label ----------
df = df[['full_text', 'label']].dropna()
df['label'] = df['label'].str.strip().map({'fake': 1, 'real': 0})

# ---------- ตรวจสอบความถูกต้อง ----------
if df['label'].nunique() < 2:
    raise ValueError("ชุดข้อมูลต้องมีทั้ง fake และ real")

# ---------- ตัดคำภาษาไทย ----------
def clean_and_tokenize(text):
    tokens = word_tokenize(text, engine='newmm')
    return ' '.join(tokens)

df['clean_text'] = df['full_text'].apply(clean_and_tokenize)

# ---------- แปลงข้อความเป็นเวกเตอร์ ----------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# ---------- แบ่งข้อมูล ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- สร้างโมเดล ----------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------- ทำนายและแสดงผล ----------
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ---------- แสดง confusion matrix ----------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import pickle

# บันทึกโมเดลและ tfidf
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# import pandas as pd
# from pythainlp.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt


# # โหลดข้อมูลจาก csv
# df = pd.read_csv('fakeNews.csv')  # สมมุติว่ามีคอลัมน์ 'text' และ 'label'
# print(df.head())

# # แปลง label เป็น 0/1
# df['label'] = df['label'].map({'fake': 1, 'real': 0})

# # ตัดคำภาษาไทย
# def clean_and_tokenize(text):
#     tokens = word_tokenize(text, engine='newmm')
#     return ' '.join(tokens)

# df['clean_text'] = df['text'].apply(clean_and_tokenize)

# # แปลงเป็นเวกเตอร์
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df['clean_text'])
# y = df['label']

# # แบ่งข้อมูล
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # สร้างโมเดล
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # ทดสอบ
# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()
