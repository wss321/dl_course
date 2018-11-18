# dl_course
面向人工智能的深度学习前沿课  实验<br/>
[实验报告文件](https://github.com/wss321/dl_course/blob/master/documents/%E5%B0%8F%E5%B0%BA%E5%AF%B8%E5%9B%BE%E7%89%87%E7%9A%84%E5%88%86%E7%B1%BB%E5%8F%8A%E9%9B%B6%E6%A0%B7%E6%9C%AC%E5%AD%A6%E4%B9%A0.pdf)<br/>
## 0.数据集
[下载数据集(需要翻墙)](https://drive.google.com/file/d/1OfqCRMZ1GBK9VaLZ5CiFDqmXVKKn5I3R/view?usp=sharing)<br/>
注:<br/>
数据集文件解压到data文件夹下<br/>
解压data下的所有压缩包<br/>
### 本实验用到了两个数据集:
1. 天池零样本学习数据集<br/>
2. AwA(Animals with Attributes)<br/>
## 1.分类
运行 code下的clf*.py 三个文件可训练得到相应的分类器<br/>
1. 超分辨:dcscn <br/>
引用：https://github.com/jiny2001/dcscn-super-resolution<br/>
超分辨结果保存在:data/classification/sr.zip<br/>
2. 模型<br/>
<img src="https://github.com/wss321/dl_course/blob/master/documents/clf.png" width="600"><br/>
3. 结果<br/>
<img src="https://github.com/wss321/dl_course/blob/master/documents/result1.png" width="600"><br/>
## 2. zero-shot learning
运行 code下的model*.py 12个文件给出对应的实验结果，在天池和AwA数据集上都进行了实验, 结果放在result文件夹下<br/>
1. 模型<br/>
<img src="https://github.com/wss321/dl_course/blob/master/documents/model1.png" width="600"><br/>
<img src="https://github.com/wss321/dl_course/blob/master/documents/model2.png" width="600"><br/>
2. 结果<br/>
<img src="https://github.com/wss321/dl_course/blob/master/documents/result2.png" width="600"><br/>
