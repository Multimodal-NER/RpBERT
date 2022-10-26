ID_TO_LABEL = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
LABEL_TO_ID = {label: idx for idx, label in enumerate(ID_TO_LABEL)}
LABEL_SET_SIZE = len(ID_TO_LABEL)
