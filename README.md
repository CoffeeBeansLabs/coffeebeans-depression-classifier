# coffeebeans-depression-classifier


This project comprises of a depression clasifier by finetuning BERT model. The model predicts two labels, where in 1 for Depressed text and 0 for Non Depressed text. 
This classification model can be extended for classifying chatbot converstion. Currently the model is trained using 20K sentences for both depression and non depression tweet data. 






```

Epoch 1/5
286/286 [==============================] - ETA: 0s - loss: 0.0372 - accuracy: 0.9886
286/286 [==============================] - 526s 2s/step - loss: 0.0372 - accuracy: 0.9886 - val_loss: 0.0176 - val_accuracy: 0.9948

Epoch 2/5
286/286 [==============================] - ETA: 0s - loss: 0.0254 - accuracy: 0.9935
286/286 [==============================] - 526s 2s/step - loss: 0.0254 - accuracy: 0.9935 - val_loss: 0.0245 - val_accuracy: 0.9926

Epoch 3/5
286/286 [==============================] - ETA: 0s - loss: 0.0178 - accuracy: 0.9956
286/286 [==============================] - 526s 2s/step - loss: 0.0178 - accuracy: 0.9956 - val_loss: 0.0195 - val_accuracy: 0.9943

Epoch 4/5
286/286 [==============================] - ETA: 0s - loss: 0.0081 - accuracy: 0.9985
286/286 [==============================] - 536s 2s/step - loss: 0.0081 - accuracy: 0.9985 - val_loss: 0.0174 - val_accuracy: 0.9965

Epoch 5/5
286/286 [==============================] - ETA: 0s - loss: 0.0046 - accuracy: 0.9989
286/286 [==============================] - 525s 2s/step - loss: 0.0046 - accuracy: 0.9989 - val_loss: 0.0183 - val_accuracy: 0.9952

``` 


