import pandas as pd
import numpy as np
from sklearn import preprocessing

# 读取文件
f_train = pd.read_csv('../bayes_data/f_train.csv', encoding='gb2312')
f_test = pd.read_csv('../bayes_data/f_test.csv', encoding='gb2312')

# 将label单独提取出来
f_train_label = f_train['label']
f_train.drop(['label'], axis=1, inplace=True)

# print(f_train)
# 数据清洗
# 删除缺失值大于一半的特征
del_feature = []
del_feature.extend(['id'])
feature = f_train.count()

drop_feature = feature[feature < 500]
del_feature.extend(drop_feature.index)
del_feature = list(set(del_feature))
f_train.drop(del_feature, axis=1, inplace=True)
f_test.drop(del_feature, axis=1, inplace=True)

# print(f_train)

# SNP为离散型变量
category_feature = ['SNP'+str(i) for i in range(1,56)]
category_feature.extend(['DM家族史', 'ACEID'])

# 得到目前还剩余的类别特征
category_feature = list(set(f_train.columns) & set(category_feature))
continuous_feature = list(set(f_train.columns) - set(category_feature))


# 填充缺失数据
# 平均值填充mean
f_train['孕前体重'] = f_train['孕前体重'].fillna(f_train['孕前体重'].mean())
f_train['孕前BMI'] = f_train['孕前BMI'].fillna(f_train['孕前BMI'].mean())
f_train['VAR00007'] = f_train['VAR00007'].fillna(f_train['VAR00007'].mean())
f_train['wbc'] = f_train['wbc'].fillna(f_train['wbc'].mean())
f_train['ALT'] = f_train['ALT'].fillna(f_train['ALT'].mean())
f_train['HDLC'] = f_train['HDLC'].fillna(f_train['HDLC'].mean())
f_train['hsCRP'] = f_train['hsCRP'].fillna(f_train['hsCRP'].mean())
f_train['TG'] = f_train['TG'].fillna(f_train['TG'].mean())

f_test['孕前体重'] = f_test['孕前体重'].fillna(f_test['孕前体重'].mean())
f_test['孕前BMI'] = f_test['孕前BMI'].fillna(f_test['孕前BMI'].mean())
f_test['VAR00007'] = f_test['VAR00007'].fillna(f_test['VAR00007'].mean())
f_test['wbc'] = f_test['wbc'].fillna(f_test['wbc'].mean())
f_test['ALT'] = f_test['ALT'].fillna(f_test['ALT'].mean())
f_test['HDLC'] = f_test['HDLC'].fillna(f_test['HDLC'].mean())
f_test['hsCRP'] = f_test['hsCRP'].fillna(f_test['hsCRP'].mean())
f_test['TG'] = f_test['TG'].fillna(f_test['TG'].mean())

# 众数填充mode
# 中位数median
f_train.fillna(f_train.median(axis=0), inplace=True)
f_test.fillna(f_train.median(axis=0), inplace=True)
print(f_train)

# 归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(f_train.values)
f_train = pd.DataFrame(data=X, columns=f_train.columns)
X = min_max_scaler.fit_transform(f_test.values)
f_test = pd.DataFrame(data=X, columns=f_test.columns)

print(f_train['wbc'])

# 对于离散型变量进行one_hot编码
train_one_hot = pd.DataFrame()
for feature in category_feature:
    feature_dummy = pd.get_dummies(f_train[feature],prefix=feature)
    f_train.drop([feature], axis=1, inplace=True)
    train_one_hot = pd.concat([train_one_hot, feature_dummy], axis=1)

test_one_hot = pd.DataFrame()
for feature in category_feature:
    feature_dummy = pd.get_dummies(f_test[feature], prefix=feature)
    f_test.drop([feature], axis=1, inplace=True)
    test_one_hot = pd.concat([test_one_hot, feature_dummy], axis=1)

# 计算相关度
def cal_corrcoef(float_df,y_train,float_col):
    corr_values = []
    for col in float_col:
        corr_values.append(abs(np.corrcoef(float_df[col].values,y_train)[0,1]))
    corr_df = pd.DataFrame({'col':float_col,'corr_value':corr_values})
    corr_df = corr_df.sort_values(by='corr_value',ascending=False)
    return corr_df

# 统计连续变量中与label的相关度大于0.01
Correlation_degree = 0.1
continuous_corr = cal_corrcoef(f_train, f_train_label, f_train.columns)
corr01 = continuous_corr[continuous_corr.corr_value >= Correlation_degree]
corr01_col = corr01['col'].values.tolist()

train_importance = pd.DataFrame()
test_importance = pd.DataFrame()
train_importance = f_train[corr01_col]
test_importance = f_test[corr01_col]

# 统计离散变量中与label的相关度大于0.1的
corr_one_hot = cal_corrcoef(train_one_hot, f_train_label, train_one_hot.columns)
corr02 = corr_one_hot[corr_one_hot.corr_value >= Correlation_degree]
corr02_col = corr02['col'].values.tolist()

train_importance = pd.concat([train_importance, train_one_hot[corr02_col]], axis=1)
test_importance = pd.concat([test_importance, test_one_hot[corr02_col]], axis=1)
# print(train_importance)
train_importance = pd.concat([train_importance, f_train_label], axis=1)


train_importance.to_csv('../bayes_data/train_important_feature.csv',encoding='gb2312')
test_importance.to_csv('../bayes_data/test_important_feature.csv',encoding='gb2312')

# print(f_train_label[0])
# print(train_importance)
# print(f_train['孕前体重'])
