# RpBERT
RpBERT: A Text-image Relationship Propagation Based BERT Model  for Multimodal NER

### Environment
#### Python packages
>- python==3.7
>- pytorch==1.2.0
>- gensim==3.3.0
>- numpy==1.15.0
>- pandas==0.20.3

**(pip install -r requirements.txt)**

### Configuration
All configuration are listed in config.py. Please verify parameters before running the codes.

### Usage
#### Training
>- python process_data.py
>- python train.py 

If you run TCSR without SRN network, please set  self.if_span_te = False in config.py.

If you run BERT for contextual network, please set  self.use_bert = True in config.py. 

#### Testing
>- python test.py

### Test Best Model:
The best model is located on "./model" path. You can change the "test_model_path" to choose model and run
"python test.py" to evaluate it.
