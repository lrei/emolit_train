"""Example of inference with these models using a huggingface pipeline."""

from transformers import pipeline

# Available models
# lrei/roberta-large-emolit
# lrei/roberta-base-emolit
# lrei/distilroberta-base-emolit
MODEL = "lrei/roberta-large-emolit"


classifier = pipeline(
    "text-classification",
    model=MODEL,
    device=None,
)

texts = ["This is so much fun!",
         "I want to go back home to my dogs, i'll be happy to go back to them."]


print()
results = classifier(texts, top_k=None, function_to_apply="sigmoid")
for t, r in zip(texts, results):
    print(t)
    print(r)
    print("\n")
