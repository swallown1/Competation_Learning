import pandas as pd
import numpy as np
from pandas import  Series,DataFrame
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocesse

def plotshow(data):
    fig = plt.figure()
    fig.set(alpha=0.2)# 设定图表颜色alpha参数

    plt.subplot2grid((2, 3), (0, 0))
    data.Survived.value_counts().plot(kind='bar')
    plt.title(u"获救人数,(1为获救)")
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 3), (0, 1))
    data.Pclass.value_counts().plot(kind='bar')
    plt.title(u"乘客等级分布")
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data.Survived, data_train.Age)
    plt.ylabel(u"年龄")  # 设定纵坐标名称
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布 (1为获救)")

    plt.subplot2grid((2,3),(1,0),colspan=2)
    data.Age[data.Pclass == 1].plot(kind = 'kde')
    data.Age[data.Pclass == 2].plot(kind = 'kde')
    data.Age[data.Pclass == 3].plot(kind = 'kde')
    plt.xlabel(u"年龄")  # plots an axis lable
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # sets our legend for our graph.

    plt.subplot2grid((2,3),(1,2))
    data.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")

    plt.show()

def set_missing_ages(df):
    # 使用sklearn 中的RandomForest来拟合缺失的数据
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    # 将乘客分为已知和未知两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    nuKnow_age = age_df[age_df.Age.isnull()].as_matrix()

    y=known_age[:,0]
    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)

    prediction = rfr.predict(nuKnow_age[:,1::])

    df.loc[(df.Age.isnull()),'Age'] = prediction

    return df , rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'

    return df

def feature_yinzi(data_train):
    # 因为逻辑回归建模时，需要输入的特征都是数值型特征，我们通常会先对类目型的特征因子化。
    # 例如 ：原本Cabin取值为yes的，在此处的"Cabin_yes"下取值为1，在"Cabin_no"下取值为0
    #         原本Cabin取值为no的，在此处的"Cabin_yes"下取值为0，在"Cabin_no"下取值为1

    dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')
    dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')

    df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis=1)
    df.drop(['Embarked','Cabin','Pclass','Sex','Name','Ticket'],axis=1,inplace=True)
    return  df

def Scaling(df):
    # 将数据中值差异较大的进行规范化，使得不会对模型造成很大的影响
    scaler = preprocesse.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)
    return df

def build_model():
    # fit到RandomForestRegressor之中
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    return clf

def data_test():
    data_test = pd.read_csv("./data/test.csv")
    data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
    # 接着我们对test_data做和train_data中一致的特征变换
    # 首先用同样的RandomForestRegressor模型填上丢失的年龄
    tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].as_matrix()
    # 根据特征属性X预测年龄并补上
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

    scaler = preprocesse.StandardScaler()
    data_test = set_Cabin_type(data_test)
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

    df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    df_test = Scaling(df_test)
    return df_test

if __name__ == '__main__':

# [891 rows x 12 columns]
# print(data_train.columns)
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch',
# 'Ticket', 'Fare', 'Cabin', 'Embarked']

    data_train = pd.read_csv('./data/train.csv')
    # print(data_train.info()) Cabin存在大量缺失值  ，Age存在部分缺失值

    # print(data_train.describe())
    # 查看每个属性的均值 方差 等信息情况

    # plotshow(data_train)

    # 简单的数据预处理
    data_train,rfr = set_missing_ages(data_train)
    data_train = set_Cabin_type(data_train)
    # print(data_train)

    # 特征因子
    data_train = feature_yinzi(data_train)
    # print(data_train)

    # 对年龄和Fare进行标准化
    df = Scaling(data_train)
    # print(data_train)

    # 用正则取出我们要的属性值
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.as_matrix()

    # y即Survival结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # 进行训练
    model = build_model()
    model.fit(X,y)

    df_test=data_test()

    # 进行预测
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = model.predict(test)
    result = DataFrame({'PassengerId':df_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("./data/logistic_regression_predictions.csv", index=False)


