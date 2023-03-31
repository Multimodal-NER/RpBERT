# RpBERT
This is a implementation of the paper 
[RpBERT: A Text-image Relation Propagation-based BERT Model for Multimodal NER](https://ojs.aaai.org/index.php/AAAI/article/view/17633).

## Requirements

### Datasets
* Download multi-modal NER dataset Twitter-15 [(Zhang et al., 2018)](http://qizhang.info/paper/aaai2017-twitterner.pdf)
from [here](http://qizhang.info/paper/data/aaai2018_multimodal_NER_data.zip)
to [this path](resources/datasets/twitter2015).
* Download multi-modal NER dataset Twitter-17 [(Lu et al., 2018)](https://aclanthology.org/P18-1185.pdf)
to [this path](resources/datasets/twitter2017).
* Download text-image relationship dataset [(Vempala et al., 2019)](https://aclanthology.org/P19-1272/)
from [here](https://github.com/danielpreotiuc/text-image-relationship) to 
[this path](resources/datasets/relationship).

Run [loader.py](data/loader.py) to make sure the statistics is identical as 
[(Zhang et al., 2018)](http://qizhang.info/paper/aaai2017-twitterner.pdf) and 
[(Lu et al., 2018)](https://aclanthology.org/P18-1185.pdf).

| Twitter-15  | NUM  | PER  | LOC  | ORG  | MISC |
| ----------- | ---- | ---- | ---- | ---- | ---- |
| Training    | 4000 | 2217 | 2091 | 928  | 940  |
| Development | 1000 | 552  | 522  | 247  | 225  |
| Testing     | 3257 | 1816 | 1697 | 839  | 726  |

| Twitter-17  | NUM   | TOKEN |
| ----------- | ----- | ----- |
| Training    | 4290  | 68655 |
| Development | 1432  | 22872 |
| Testing     | 1459  | 23051 |

### Models
* Download pre-trained ResNet-101 weights
from [here](https://download.pytorch.org/models/resnet101-63fe2227.pth)
to [this path](resources/models/cnn/resnet101.pth).
* Download pre-trained BERT-Base weights 
from [here](https://huggingface.co/bert-base-uncased/tree/main)
to [this path](resources/models/transformers/bert-base-uncased).
* Download pre-trained word embeddings
from [here](https://flair.informatik.hu-berlin.de/resources/embeddings/token/)
to [this path](resources/models/embeddings).

### Libraries
* tqdm
* Pillow
* numpy
* torch
* torchvision
* transformers
* flair
* pytorch-crf

## Usage

```shell script
# BERT-BiLSTM-CRF
python main.py --stacked --rnn --crf --dataset [dataset_id] --cuda [gpu_id]
# RpBERT-BiLSTM-CRF
python main.py --stacked --rnn --crf --encoder_v resnet101 --aux --gate --dataset [dataset_id] --cuda [gpu_id]
```
