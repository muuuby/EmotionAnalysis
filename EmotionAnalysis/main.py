# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import numpy as np

from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

#数据预处理
a = []  #读取数据文件
fp=open("datafiles/train_1.txt")
for line in fp.readlines():
    line=line.replace('\n','')  #去除每一行最后的换行符号
    line=line.split('\t')   #以制表符为分隔符
    a.append(line)               
fp.close()
df = pd.DataFrame(a)    #将数据转换为df格式
df.columns = ['stars', 'text']
df[['stars']] = df[['stars']].apply(pd.to_numeric)  #将stars这一列转化为数字类型

def make_label(df): #大于三星的标签为p，小于三星的标签为n
    df["sentiment"] = df["stars"].apply(lambda x: 'p' if x>3 else 'n')
make_label(df)

x = df[['text']]    #评论内容
y = df.iloc[:,2]  #分类标签

#特征向量化
def word_cut(text): #使用jieba进行分词
    return " ".join(jieba.cut(text))
x['cut_text'] = x.text.apply(word_cut)
#设定参数进行更好的特征选择
best_min_df = 0.001  #当某个词出现的词频小于min_df，这个词不会被当作关键词
best_score = 0
vect = CountVectorizer(min_df=best_min_df,
                       token_pattern='(?u)\\b[^\\d\\W]\\w+\\b',    
                       stop_words='english')    #去除英文文本的停用词、去除数字
#进行向量转换
term_matrix = pd.DataFrame(vect.fit_transform(x.cut_text).toarray(), columns=vect.get_feature_names())
print (term_matrix)

nb = MultinomialNB()    #使用朴素贝叶斯模型
pipe = make_pipeline(vect, nb)

pipe.fit(x.cut_text,y)  #使用训练集训练设定好参数的模型

#读取测试文本
b = []  #读取数据文件
fp=open("datafiles/test_sample_1.txt")
for line in fp.readlines():
    line=line.replace('\n','')  #去除每一行最后的换行符号
    line=line.split('\t')   #以制表符为分隔符
    b.append(line)               
fp.close()
z = pd.DataFrame(b)    #将数据转换为df格式
z.columns = ['text']
z['cut_text'] = z.text.apply(word_cut)  

z_predict = pipe.predict(z.cut_text)  #获得预测结果
f = open(u'datafiles/result.txt','w') #将结果保存为文件
for i in z_predict:
    f.write(str(i)+'\n')  
f.close()

