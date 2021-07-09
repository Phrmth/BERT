! pip install transformers # installing transformers from hugging face

from transformers import Pipeline 
classifier = pipeline("sentiment-analysis") # selecting the model based on the task we want to perform, Ex - summarization, text classification, text generation

classifier('This is awesome book") # Testing our classifier output
           
           
           #** more to come**
