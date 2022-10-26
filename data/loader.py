import re
import csv
from pathlib import Path
from collections import Counter
from data.dataset import MyToken, MySentence, MyImage, MyPair, MyDataset, MyCorpus
import constants


# constants for preprocessing
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
IMGID_PREFIX = 'IMGID:'
URL_PREFIX = 'http://t.co/'
UNKNOWN_TOKEN = '[UNK]'


def normalize_text(text: str):
    # remove the ending URL which is not part of the text
    url_re = r' http[s]?://t.co/\w+$'
    text = re.sub(url_re, '', text)
    return text


def load_itr_corpus(path: str, split: int = 3576, normalize: bool = True):
    path = Path(path)
    path_to_images = path / 'images'
    assert path.exists()
    assert path_to_images.exists()

    with open(path/'data.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, doublequote=False, escapechar='\\')
        pairs = [MyPair(
            sentence=MySentence(text=normalize_text(row['tweet']) if normalize else row['tweet']),
            image=MyImage(f"T{row['tweet_id']}.jpg"),
            label=int(row['image_adds'])
        ) for row in csv_reader]

    train = MyDataset(pairs[:split], path_to_images)
    test = MyDataset(pairs[split:], path_to_images)
    return MyCorpus(train=train, test=test)


def load_ner_dataset(path_to_txt: Path, path_to_images: Path, load_image: bool = True) -> MyDataset:
    tokens = []
    image_id = None
    pairs = []

    with open(str(path_to_txt), encoding='utf-8') as txt_file:
        for line in txt_file:
            line = line.rstrip()  # strip '\n'

            if line.startswith(IMGID_PREFIX):
                image_id = line[len(IMGID_PREFIX):]
            elif line != '':
                text, label = line.split('\t')
                if text == '' or text.isspace() \
                        or text in SPECIAL_TOKENS \
                        or text.startswith(URL_PREFIX):
                    text = UNKNOWN_TOKEN
                tokens.append(MyToken(text, constants.LABEL_TO_ID[label]))
            else:
                pairs.append(MyPair(MySentence(tokens), MyImage(f'{image_id}.jpg')))
                tokens = []
    pairs.append(MyPair(MySentence(tokens), MyImage(f'{image_id}.jpg')))

    return MyDataset(pairs, path_to_images, load_image)


def load_ner_corpus(path: str, load_image: bool = True) -> MyCorpus:
    path = Path(path)
    path_to_train_file = path / 'train.txt'
    path_to_dev_file = path / 'dev.txt'
    path_to_test_file = path / 'test.txt'
    path_to_images = path / 'images'

    assert path_to_train_file.exists()
    assert path_to_dev_file.exists()
    assert path_to_test_file.exists()
    assert path_to_images.exists()

    train = load_ner_dataset(path_to_train_file, path_to_images, load_image)
    dev = load_ner_dataset(path_to_dev_file, path_to_images, load_image)
    test = load_ner_dataset(path_to_test_file, path_to_images, load_image)

    return MyCorpus(train, dev, test)


def type_count(dataset: MyDataset) -> str:
    tags = [token.label for pair in dataset for token in pair.sentence]
    counter = Counter(tags)

    num_total = len(dataset)
    num_per = counter['B-PER']
    num_loc = counter['B-LOC']
    num_org = counter['B-ORG']
    num_misc = counter['B-MISC']

    return f'{num_total}\t{num_per}\t{num_loc}\t{num_org}\t{num_misc}'


def token_count(dataset: MyDataset) -> str:
    lengths = [len(pair.sentence) for pair in dataset]

    num_sentences = len(lengths)
    num_tokens = sum(lengths)

    return f'{num_sentences}\t{num_tokens}'


if __name__ == "__main__":
    twitter2015 = load_ner_corpus('resources/datasets/twitter2015')
    twitter2015_train_statistic = type_count(twitter2015.train)
    twitter2015_dev_statistic = type_count(twitter2015.dev)
    twitter2015_test_statistic = type_count(twitter2015.test)
    assert twitter2015_train_statistic == '4000\t2217\t2091\t928\t940'
    assert twitter2015_dev_statistic == '1000\t552\t522\t247\t225'
    assert twitter2015_test_statistic == '3257\t1816\t1697\t839\t726'

    print('-----------------------------------------------')
    print('2015\tNUM\tPER\tLOC\tORG\tMISC')
    print('-----------------------------------------------')
    print('TRAIN\t' + twitter2015_train_statistic)
    print('DEV\t' + twitter2015_dev_statistic)
    print('TEST\t' + twitter2015_test_statistic)
    print('-----------------------------------------------')

    print()

    twitter2017 = load_ner_corpus('resources/datasets/twitter2017')
    twitter2017_train_statistic = token_count(twitter2017.train)
    twitter2017_dev_statistic = token_count(twitter2017.dev)
    twitter2017_test_statistic = token_count(twitter2017.test)
    assert twitter2017_train_statistic == '4290\t68655'
    assert twitter2017_dev_statistic == '1432\t22872'
    assert twitter2017_test_statistic == '1459\t23051'

    print('------------------------')
    print('2017\tSENT.\tTOKEN')
    print('------------------------')
    print('TRAIN\t' + twitter2017_train_statistic)
    print('DEV\t' + twitter2017_dev_statistic)
    print('TEST\t' + twitter2017_test_statistic)
    print('------------------------')
