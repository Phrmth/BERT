# Downloading BERT pretrained model and testing the text classfication on the input passed.

! pip install transformers # installing transformers from hugging face
from transformers import Pipeline 

classifier = pipeline("sentiment-analysis") # selecting the model based on the task we want to perform, Ex - summarization, text classification, text generation
classifier('This is awesome book") # Testing our classifier output
           

           
 # downlaodng the multilingual model for classification
classifier = pipeline('sentiment-analysis', model = 'nlptown/bert-base-multilingual-uncased-sentiment')      
classifier("le film était pathétique"). # a list of sentenses can be passed in as well         
           

# Using BERT pre-trained for classification task on a different dataset
           
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow_datasets
import tensorflow as tf
from transformers import glue_convert_examples_to_features

           
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
           
data = tensorflow_datasets.load('glue/mrpc')
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length = 128, task = 'mrpc')
test_dataset = glue_convert_examples_to_features(data['test'], tokenizer, max_length = 128, task = 'mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
test_dataset = test_dataset.batch(64)
           
           

optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-5, epsilon = 1e-08, clipnorm = 1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss = loss, metrics=[metrics])
           
           
model.fit(train_dataset, epochs = 2, validation_data=test_dataset, validation_steps=7)
