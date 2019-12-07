# Neural Machine Translation in OpenCL

In this project, we trained a LSTM network which converts the English text to German text
based on its vocabulary. For this we choose an encoder-decoder architecture variant. The
encoder is a LSTM layer that takes the input of latent dimension of 64.

# Dataset details:
Dataset source: http://www.manythings.org/anki/

There are over 700,000 English sentences. Over half of these (400,000+) have audio files. It is an
extensive dataset with over 30+ different bilingual datasets.

Trained for around 250 epochs   

Number of clean samples: 7156  
Number of unique input tokens: 71  
Number of unique output tokens: 84   
Max sequence length for inputs: 16  
Max sequence length for outputs: 45  
Train on 5724 samples, validate on 1432 samples

# OpenCL implementation:

Baseline Keras implementation: 85.57%  

Using standard Sigmoid function:  
Average time taken by program on device : 0.042016 sec  
Accuracy: 73.2317%

Using sigmoid approximation:  
Time taken by program on device : 0.038085 sec  
Accuracy: 64.1302%

For more details, refer to the project report.
