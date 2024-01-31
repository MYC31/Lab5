import torch
import torch.nn as nn
from torch.nn import LSTM
from transformers import BertModel
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AblationTxTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.t_linear = nn.Linear(768, 256)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:, 0, :]
        txt_out.view(txt_out.shape[0], -1)
        txt_out = self.t_linear(txt_out)
        txt_out = self.relu(txt_out)
        last_out = self.fc(txt_out)
        return last_out


class AblationImgModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_model = torchvision.models.resnet18(pretrained=True)
        self.i_linear = nn.Linear(1000, 256)
        self.fc = nn.Linear(256, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        img_out = self.img_model(image)
        img_out = self.i_linear(img_out)
        img_out = self.relu(img_out)
        last_out = self.fc(img_out)
        return last_out


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.img_model = torchvision.models.resnet34(pretrained=True)
        self.t_linear = nn.Linear(768, 128)
        self.i_linear = nn.Linear(1000, 128)
        self.img_q = nn.Linear(128, 1)
        self.txt_q = nn.Linear(128, 1)
        self.fc = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        img_out = self.img_model(image)
        img_out = self.i_linear(img_out)
        img_out = self.relu(img_out)
        img_weight = self.img_q(img_out)

        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:, 0, :]
        txt_out.view(txt_out.shape[0], -1)
        txt_out = self.t_linear(txt_out)
        txt_out = self.relu(txt_out)
        txt_weight = self.txt_q(txt_out)

        last_out = img_weight * img_out + txt_weight * txt_out
        last_out = self.fc(last_out)
        return last_out


class BiLSTMModel(nn.Module):
    def __init__(self):
        super(BiLSTMModel, self).__init__()
        hidden_size = 128
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.img_model = torchvision.models.resnet18(pretrained=True)
        self.t_linear = nn.Linear(768, 128)
        self.i_linear = nn.Linear(1000, 2*128)
        self.bilstm = LSTM(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.img_q = nn.Linear(2*128, 1)
        self.txt_q = nn.Linear(128, 1)
        self.fc = nn.Linear(2 * hidden_size, 3)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        img_out = self.img_model(image)
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        img_out = self.i_linear(img_out)
        txt_out = self.t_linear(txt_out)
        img_weight = self.img_q(img_out)
        txt_weight = self.txt_q(txt_out)
        lstm_out, _ = self.bilstm(txt_out.unsqueeze(1))
        lstm_out = lstm_out[:, -1, :]
        fusion = img_weight * img_out + txt_weight * lstm_out
        return self.fc(self.relu(fusion))


def train_process(model, epoch_num, optimizer, train_dataloader, valid_dataloader, train_count, valid_count):
    Loss_C = nn.CrossEntropyLoss()
    train_acc = []
    valid_acc = []
    for epoch in range(epoch_num):
        loss = 0.0
        train_cor_count = 0
        valid_cor_count = 0
        for b_idx, (img, des, target, idx, mask) in enumerate(train_dataloader):
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            optimizer.zero_grad()
            loss = Loss_C(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
            train_cor_count += int(pred.eq(target).sum())
        train_acc.append(train_cor_count / train_count)
        for img, des, target, idx, mask in valid_dataloader:
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            pred = output.argmax(dim=1)
            valid_cor_count += int(pred.eq(target).sum())
        valid_acc.append(valid_cor_count / valid_count)
        print('Train Epoch: {}, Train_Loss: {:.4f}, Train Accuracy: {:.4f}, Valid Accuracy: {:.4f}'.format(epoch + 1,
                                                                                                           loss.item(),
                                                                                                           train_cor_count / train_count,
                                                                                                           valid_cor_count / valid_count))
    plt.plot(train_acc, label="train_accuracy")
    plt.plot(valid_acc, label="valid_accuracy")
    plt.title(model.__class__.__name__)
    plt.xlabel("Epoch")
    plt.xticks(range(epoch_num), range(1, epoch_num + 1))
    plt.ylabel("Accuracy")
    plt.ylim(ymin=0, ymax=1)
    plt.legend()
    plt.show()
