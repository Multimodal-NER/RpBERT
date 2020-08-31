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
### word embeddings download links
>- ACE04/05 : [glove 100d](https://drive.google.com/open?id=1qDmFF0bUKHt5GpANj7jCUmDXgq50QJKw)
>- GENIA : [wikipedia-pubmed-and-PMC-w2v 200d](http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin)

Before running the code, please put word embeddings at the directory "/model/word2vec/".

### Data format

R7 - 57 reporter cells , on the other hand , signaled induced activity of the lytic origin of EBV replication ( ori Lyt ) .

NN NN NN NN NNS , IN DT JJ NN , VBD VBN NN IN DT JJ NN IN NN NN ( NN NN ) .

0,5 G#cell_line|16,21 G#DNA|22,24 G#DNA

The first line is a sentence. The second line is POS tags. The third line is the location (start,end] and type of entity separated by "|". For example, "0,5 G#cell_line" denotes "R7 - 57 reporter cells"  is a "cell_line".


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
