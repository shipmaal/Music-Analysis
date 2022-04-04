from transformers import pipeline
from pprint import pprint

classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-go-emotion',
                      return_all_scores=True)

prediction = classifier("The happy baby giggled")
pprint(prediction)

# for i in prediction[0]:
#     if i['score'] > 0.1:
#         print(i)