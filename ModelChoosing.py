from transformers import pipeline

review_text = "Can't say enough good things. The fit, the material quality, the price- all perfect. I ordered oversized, an XL, in the Heather gray color, it's exactly what I wanted. Cool girl vibes lol fits just right. I'm 5'5 and 170 lbs for reference. The fabric literally feels like this should cost more, it's nice. Also, this sweater is cozy and will definitely keep you warm. Win win all-around!"

classifier = pipeline("sentiment-analysis")
print("Default: ")
print(classifier(review_text))

classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
print("BERT: ")
print(classifier(review_text))

classifier = pipeline("sentiment-analysis", model="LiYuan/amazon-review-sentiment-analysis")
print("Amazon Review Pipeline: ")
print(classifier(review_text))
