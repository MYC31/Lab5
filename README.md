## 依赖安装

通过运行以下代码安装本项目所需依赖。

```python
pip install -r requirements.txt
```



## 文件树结构

以下是一些重要文件及其描述。

```python
|-- data # 图片和文本数据
|-- report.pdf # 项目实验报告
|-- models.py # 包含了所有实现的模型
|-- main.py # 模型训练及预测过程
|-- test.txt # 预测输出文件
|-- requirement.txt # 运行所需依赖
|-- train.txt # 训练数据
|-- test_with_label.txt # 需要预测的文件
```



##  代码运行

可选参数为：

`--model` ：可选fusion、lstm

`--image_only`：只输入图片，不输入默认为false，与text_only参数互斥

`--text_only`：只输入文字，不输入默认为false

 `--lr`：初始学习率，不输入默认为1e-5

`--epoch_num`：训练迭代次数，默认为10次

例如：1.在命令行中输入以下代码，即可选用fusion动态加权的模型，学习率为1e-6，迭代15次

```
python main.py --model fusion --lr 1e-6 --epoch_num 15
```

2.在命令行中输入以下代码，即可运行只输入图片的消融实验，学习率为1e-5，迭代10次

```
python main.py --image_only --model bert_resnet_weight
```



### 参考

无
