from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from models import device


def embeddings(txt, token):
    result = token.batch_encode_plus(batch_text_or_text_pairs=txt, truncation=True, padding='max_length', max_length=32,
                                     return_tensors='pt')
    input_ids = result['input_ids']
    attention_mask = result['attention_mask']
    return input_ids, attention_mask


class NewDataset():
    def __init__(self, images, descriptions, tags, token):
        self.images = images
        self.descriptions = descriptions
        self.tags = tags
        self.input_ids, self.attention_masks = embeddings(self.descriptions, token)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        img = self.images[idx]
        des = self.descriptions[idx]
        tag = self.tags[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        return img, des, tag, input_id, attention_mask


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


def save_predicted_value(model, token, args):
    emo_list = ["neutral", "negative", "positive"]
    test_df = pd.read_csv("./test_without_label.txt")
    guid_list = test_df['guid'].tolist()
    tag_pre_list = []

    for idx in guid_list:
        img = Image.open('./data/' + str(idx) + '.jpg')
        img = img.resize((224,224), Image.Resampling.LANCZOS)
        image = np.asarray(img, dtype = 'float32')
        image = image.transpose(2, 0, 1)
        with open('./data/' + str(idx) + '.txt', encoding='gb18030') as fp:
            description = fp.read()
        input_id, mask = embeddings([description],token)
        image = image.reshape(1,image.shape[0], image.shape[1], image.shape[2])
        y_pred = model(input_id.to(device), mask.to(device), torch.Tensor(image).to(device))
        tag_pre_list.append(emo_list[y_pred[0].argmax(dim=-1).item()])
    
    result_df = pd.DataFrame({'guid':guid_list, 'tag':tag_pre_list})

    des_file = None
    if args.image_only:
        des_file = "my_test_img.txt"
    elif args.text_only:
        des_file = "my_test_txt.txt"
    else:
        if (args.model == 'fusion'):
            des_file = "my_test_fusion.txt"
        elif (args.model == 'lstm'):
            des_file = "my_test_lstm.txt"

    result_df.to_csv(des_file, sep=',', index=False)


def get_dataloader(img_txt_pairs, tags, token):
    X_train, X_valid, tag_train, tag_valid = train_test_split(img_txt_pairs, tags, test_size=0.2, random_state=1458, shuffle=True)
    image_train, txt_train = [X_train[i][0] for i in range(len(X_train))], [X_train[i][1] for i in range(len(X_train))]
    image_valid, txt_valid = [X_valid[i][0] for i in range(len(X_valid))], [X_valid[i][1] for i in range(len(X_valid))]
    train_dataset = NewDataset(image_train, txt_train, tag_train, token)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataset = NewDataset(image_valid, txt_valid, tag_valid, token)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)
    return train_dataloader, valid_dataloader, {"x_train": len(X_train), "x_valid": len(X_valid)}

