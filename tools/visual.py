import torch
import numpy as np
from flair.data import Sentence
from pytorch_pretrained_bert import BertTokenizer
from typing import List, Union, Dict, Tuple
import matplotlib.pyplot as plt


class BertInputFeatures(object):
    """Private helper class for holding BERT-formatted features"""

    def __init__(
            self,
            unique_id,
            tokens,
            input_ids,
            input_mask,
            input_type_ids,
            token_subtoken_count,
    ):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.token_subtoken_count = token_subtoken_count


def visual_attenion(text, embedding, img_name):
    sentence = Sentence(text)
    # print(sentence)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bert_tokenization: List[str] = []
    token_subtoken_count: Dict[int, int] = {}

    for token in sentence:
        subtokens = tokenizer.tokenize(token.text)
        bert_tokenization.extend(subtokens)
        token_subtoken_count[token.idx] = len(subtokens)

    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in bert_tokenization:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.

    feature = BertInputFeatures(
        unique_id=0,
        tokens=tokens,
        input_ids=input_ids,
        input_mask=input_mask,
        input_type_ids=input_type_ids,
        token_subtoken_count=token_subtoken_count,
    )

    print(tokens)
    print(embedding)
    # embedding = [0.0010679488768801093, 0.0007235786179080606, 0.0006705399136990309, 0.0005758945480920374,
    #              0.0005897579831071198, 0.0008165386971086264, 0.0006056257407180965, 0.0006978624151088297,
    #              0.0011979432310909033, 0.0009668002603575587, 0.0008826295961625874, 0.0011053975904360414,
    #              0.0012109082890674472, 0.000861537002492696]

    subtoken_embeddings = []
    for token_index, _ in enumerate(feature.tokens):
        # all_layers = []
        # for layer_index in self.layer_indexes:
        #     layer_output = all_encoder_layers[int(layer_index)][
        #         sentence_index
        #     ]
        #     all_layers.append(layer_output[token_index])

        subtoken_embeddings.append(embedding[token_index])
    # print(len(subtoken_embeddings))
    # get the current sentence object
    token_idx = 0
    attention_list = []
    for token in sentence:
        # add concatenated embedding to sentence
        token_idx += 1
        attns = subtoken_embeddings[
                     token_idx: token_idx
                                + feature.token_subtoken_count[token.idx]
                     ]
        attn_sum = 0
        for item in attns:
            attn_sum += item
        attention_list.append(attn_sum)
        token_idx += feature.token_subtoken_count[token.idx] - 1

    # print(attention_list)

    attention = np.asarray(attention_list[:-3])
    attention = np.expand_dims(attention, axis=0)
    # print(attention)


    plt.matshow(attention, vmin=0.00018139391613658518, vmax=0.0007103198440745473)
    plt.xticks(range(len(attention_list[:-3])), text.split(' ')[:-3], fontsize=8)
    plt.colorbar()
    plt.savefig(img_name)


text = 'Go hiking w/a Viking ! PSD Lesher teachers take school spirit to top of 14ner Mount Sherman . #fallfun #Coloradoproud http://t.co/nSktiRQw6K'
embedding = [0.002702075522392988, 0.0008485878352075815, 0.00025809515500441194, 0.0005950687918812037, 0.0007746960036456585, 0.0005656896391883492, 0.00011044392886105925, 0.0010470132110640407, 0.00012764472921844572, 9.752006008056924e-05, 0.00016779889119789004, 0.00011744294170057401, 0.0006886375485919416, 0.0008640392334200442, 0.0002460908726789057, 0.00022756136604584754, 0.0005689060199074447, 0.0003518823941703886, 0.00038378650788217783, 0.00029064048430882394, 0.0002536565298214555, 9.00812228792347e-05, 9.902690362650901e-05, 0.0012361527187749743, 0.0005038469098508358, 0.00027122042956762016, 0.00026482235989533365, 0.0006660179351456463, 0.00046832190128043294, 0.00011676908616209403, 0.0002815103798639029, 0.0004981849924661219, 0.001518572447821498, 0.0016285779420286417, 0.0018218545010313392, 0.0016668379539623857, 0.0014544578734785318, 0.0018343421397730708, 0.0018519642762839794, 0.001992587698623538, 0.0009838308906182647, 0.000880754494573921, 0.0013246906455606222, 0.0015885925386101007, 0.0017489492893218994, 0.0019052199786528945, 0.002055202145129442, 0.000813037040643394]


visual_attenion(text, embedding, 'text_visual.jpg')

"""
Looking forward to editing some SBU baseball shots from Saturday .
[0.0010679488768801093, 0.0007235786179080606, 0.0006705399136990309, 0.0005758945480920374, 0.0005897579831071198, 0.0008165386971086264, 0.0006056257407180965, 0.0006978624151088297, 0.0011979432310909033, 0.0009668002603575587, 0.0008826295961625874, 0.0011053975904360414, 0.0012109082890674472, 0.000861537002492696]
"""

"""
Nice image of Kevin Love and Kyle Korver during 1 st half # NBAFinals # Cavsin9 # Cleveland
[0.0012623380171135068, 0.000763462798204273, 0.0005651917308568954, 0.0007842950872145593, 0.0005853123730048537, 0.0003696624480653554, 0.0005385156255215406, 0.00046563916839659214, 0.0003195148310624063, 0.0003298553638160229, 0.0004048346891067922, 0.0010031684068962932, 0.0006007662741467357, 0.0005758404731750488, 0.0007628606981597841, 0.0009150170371867716, 0.0009552044793963432, 0.0006774549838155508, 0.0006554989377036691, 0.001012627501040697, 0.0006825783639214933, 0.000622740073595196, 0.0007125146803446114, 0.0009930829983204603, 0.0014812164008617401, 0.0013692211359739304, 0.0008943708962760866]
"""

"""
RT @verge : Reddit needs to stop pretending racism is valuable debate , by @thedextriarchy http://t.co/RMwCwoVYGD http://t.co/GK0O1pL8Sd
[0.0008071134216152132, 0.00032649602508172393, 0.0001478629419580102, 0.00022060122864786536, 0.0004214039072394371, 0.0003355613735038787, 0.0003007191407959908, 0.00039327607373706996, 0.0003214698226656765, 0.0002973658556584269, 0.00017922055849339813, 0.00019395614799577743, 0.00025560977519489825, 0.00012315269850660115, 0.00016551162116229534, 0.00039943831507116556, 0.0002730767009779811, 0.00022361769515555352, 0.0002065306471195072, 0.0001976413041120395, 0.00018898790585808456, 0.00016103274538181722, 0.00024139549350365996, 0.0004414958239067346, 0.00027084871544502676, 0.00018664836534298956, 0.000152140055433847, 0.00023803205112926662, 0.00029744970379397273, 0.0004239759291522205, 0.00021775781351607293, 0.0002705191436689347, 0.0002556458639446646, 0.0002349453861825168, 0.00029838067712262273, 0.00017543879221193492, 0.0002469008613843471, 0.0004932117881253362, 0.00033929062192328274, 0.00024950108490884304, 0.00022507771791424602, 0.0003370527701918036, 0.0004029783303849399, 0.0005597792332991958, 0.00033885231823660433, 0.00033297998015768826, 0.0003337213129270822, 0.00036158179864287376, 0.0005584650207310915, 0.0003417437255848199, 0.0004267586045898497, 0.0004863505600951612, 0.0008912640041671693, 0.000800020236056298]
"""

"""
Go hiking w/a Viking ! PSD Lesher teachers take school spirit to top of 14ner Mount Sherman . #fallfun #Coloradoproud http://t.co/nSktiRQw6K
[0.0009464335744269192, 0.0003242323000449687, 0.00018139391613658518, 0.0002450590836815536, 0.0001773570547811687, 0.00025553867453709245, 0.00027668930124491453, 0.00040386777254752815, 0.00025426375214010477, 0.00022865617938805372, 0.00039387395372614264, 0.00024754658807069063, 0.0003502877370920032, 0.0002794170577544719, 0.0002915844670496881, 0.0002476606168784201, 0.0002613357501104474, 0.0002756559697445482, 0.000302619970170781, 0.0003182604268658906, 0.00039205941720865667, 0.0004770930390805006, 0.00039462142740376294, 0.0006032164674252272, 0.0003021692973561585, 0.000497605127748102, 0.0003242356760893017, 0.0003052606771234423, 0.0002954420051537454, 0.00044189183972775936, 0.0003342539712321013, 0.0003341691044624895, 0.0005799912032671273, 0.00043927039951086044, 0.0003038387803826481, 0.00024850404588505626, 0.0004081210936419666, 0.00047228721086867154, 0.0005430169403553009, 0.00039410433964803815, 0.0004607493756338954, 0.0003078648878727108, 0.0003982977650593966, 0.00036703661317005754, 0.0004555318446364254, 0.0005474664503708482, 0.0006304126000031829, 0.000745957309845835]
"""

"""
RT @ThatDudeMCFLY : Ask Siri what 0 divided by 0 is and watch her put you in your place . http://t.co/qN1KX8YTVp
[0.0010973808821290731, 0.0004155634087510407, 0.0002944575098808855, 0.00035729145747609437, 0.0002740429190453142, 0.00032934328191913664, 0.00022789594368077815, 0.00023939127277117223, 0.000595897319726646, 0.0003838370321318507, 0.0011928975582122803, 0.001389330136589706, 0.00028146474505774677, 0.0006694339099340141, 0.00022174617333803326, 0.00018251276924274862, 0.0006680404767394066, 0.0003596318420022726, 0.00048697396414354444, 0.0006889631622470915, 0.0005399654037319124, 0.0003952266415581107, 0.0006512498366646469, 0.00026694900589063764, 0.0003944460768252611, 0.00043542933417484164, 0.000572141376323998, 0.0007168255397118628, 0.0005369246355257928, 0.0003615907044149935, 0.00029764583450742066, 0.0005242900806479156, 0.0005751153803430498, 0.0007822507177479565, 0.00033991728560067713, 0.0004060897044837475, 0.00042619940359145403, 0.00033215340226888657, 0.00031765244784764946, 0.00046751275658607483, 0.00036446930607780814, 0.00046058956650085747, 0.00046545820077881217, 0.0005767050315625966, 0.0008714019204489887]
"""