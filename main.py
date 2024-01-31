import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from transformers import BertTokenizer
import argparse
from models import FusionModel
from models import AblationTxTModel
from models import AblationImgModel
from models import BiLSTMModel
from models import train_process, device
from dataset import save_predicted_value, get_dataloader



def get_model(args):
    model = None
    if args.image_only:
        model = AblationImgModel().to(device)
    elif args.text_only:
        model = AblationTxTModel().to(device)
    else:
        if (args.model == 'fusion'):
            model = FusionModel().to(device)
        elif (args.model == 'lstm'):
            model = BiLSTMModel().to(device)
    return model


def data_preproc(descriptions):
    for i in range(len(descriptions)):
        des = descriptions[i]
        word_list = des.replace("#", "").split(" ")
        words_result = []
        for word in word_list:
            if len(word) < 1:
                continue
            elif (len(word)>=4 and 'http' in word) or word[0]=='@':
                continue
            else:
                words_result.append(word)
        descriptions[i] = " ".join(words_result)
    return descriptions


def get_data(images, descriptions, tags, emo_tag, train_dataframe):
    for i in range(train_dataframe.shape[0]):
        guid = train_dataframe.iloc[i]['guid']
        tag = train_dataframe.iloc[i]['tag']
        img = Image.open('./data/' + str(guid) + '.jpg')
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(img, dtype='float32')
        with open('./data/' + str(guid) + '.txt', encoding='gb18030') as f:
            des = f.read()
        images.append(img.transpose(2, 0, 1))
        descriptions.append(des)
        tags.append(emo_tag[tag])


def main():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--model', default='bert_resnet_simple')
    parser.add_argument('--image_only', action='store_true')
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epoch_num', default=10, type=int)
    args = parser.parse_args()

    model = get_model(args)
    lr = args.lr
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    epoch_num = args.epoch_num
    images, descriptions, tags = [], [], []
    emo_tag = {"neutral": 0, "negative": 1, "positive": 2}
    train_dataframe = pd.read_csv("./train.txt")
    pre_trained_model = "bert-base-uncased"
    token = BertTokenizer.from_pretrained(pre_trained_model, mirror='bfsu')

    get_data(images, descriptions, tags, emo_tag, train_dataframe)
    descriptions = data_preproc(descriptions)
    img_txt_pairs = [(images[i], descriptions[i]) for i in range(len(descriptions))]

    train_dataloader, valid_dataloader, size = get_dataloader(img_txt_pairs, tags, token)

    train_process(model, epoch_num, optimizer, train_dataloader, valid_dataloader, size["x_train"], size["x_valid"])

    save_predicted_value(model, token, args)


if __name__ == '__main__':
    main()
