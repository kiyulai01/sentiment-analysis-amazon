from transformers import pipeline
import csv

count = 0
review_list = []
star5 = 0
star4 = 0
star3 = 0
star2 = 0
star1 = 0

def csv_to_2d_array(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

csv_file = 'AmazonReview.csv'
reviews = csv_to_2d_array(csv_file)
classifier = pipeline("text-classification", model="LiYuan/amazon-review-sentiment-analysis")

for review in reviews:
    review_list.append(classifier(review))
    count += 1

for l in review_list:
    for d in l:
        label = d.get('label')
        if label == '5 stars':
            star5 += 1
        elif label == '4 stars':
            star4 += 1
        elif label == '3 stars':
            star3 += 1
        elif label == '2 stars':
            star2 += 1
        else:
            star1 += 1

print("5 stars: ", star5/count)
print("4 stars: ", star4/count)
print("3 stars: ", star3/count)
print("2 stars: ", star2/count)
print("1 star: ", star1/count)

if star5 == max (star5, star4, star3, star2, star1):
    print("5 stars product! Customers like it!")
elif star4 == max (star5, star4, star3, star2, star1):
    print("4 stars product! Keep going!")
elif star3 == max (star5, star4, star3, star2, star1):
    print("3 stars product! Try to improve it!")
elif star2 == max (star5, star4, star3, star2, star1):
    print("2 stars product! Need to address customers' concerm.")
else:
    print("1 stars product! Customers don't like it.")

