from sklearn.feature_extraction.text import CountVectorizer


texts = ["I love this movie", "This film is terrible", "I enjoyed the film", "I hate this movie"]

# ایجاد نمونه CountVectorizer
vectorizer = CountVectorizer()

# آموزش و تبدیل متن به وکتورهای عددی
X = vectorizer.fit_transform(texts)

# نمایش ویژگی‌ها (کلمات)
print(vectorizer.get_feature_names_out())

# نمایش وکتورهای عددی
print(X.toarray())
