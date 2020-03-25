##　Task3 特征工程

----


>day3:今天进行的是特征工程部分，也就是对一些特征进行处理，构造适合各种模型的数据。

### 特征工程的目标
对于特征进行进一步分析，并对于数据进行处理

完成对于特征工程的分析，并对于数据进行一些图表或者文字总结并打卡。

## 数据预处理
 - 空值处理：数据清洗包括，total_rooms 用中位数代替，ocean_proximity 用one-hot-encode编码转为数值型，one-hot-encode与直接编码为 [0,1,2,3..] 相比，不会引入相似或者相近的意义。比如 2 和 3 在数值上相近，但是它们各自表示的NEAR BAY与INLAND属性项并不存在实质的相似关系（尽管有这个可能）。

 - 构造特征：数据集里因为不同district里户数不同，所以 total_rooms、total_bedrooms、population 这些属性可能没有太大的意义，而每户的房间数、卧室数、人数相关性则较，所有在数据集中可以新增这些特征。

 - 归一化：数值型特征的一个特点是，不同的属性取值范围差别极大，对模型学习、正则项设置不利。有两种简单的方法可以应对这种情况：变换到0-1区间的归一化或者标准正态分布。前者对异常点非常敏感，如果存在某个值比正常值区间距离很远，这种处理会导致正常数据点分布过于集中；后者对过于集中的数据分布敏感，如果原数据集方差很小，除以近0的数值容易导致精度失真。

 - 偏度处理：在大部分数值型特征中，通常分布不符合正态分布，而是类似于“长尾”。例如房价中，低价占大部分，豪宅属于小部分。应对这种数据分布，一般可以通过神奇的log化处理转化类近似正态分布

## 异常值处理：
对于不同的特征，我们会用箱形图或者小提琴图可视化数据的分布情况。如果数据是满足正态分布的话，我们还可以按照3δ准则，将3δ以外的数据去除。本次学习中去除的方法是将数据的25%-75%范围扩展box_scale倍，在此范围以外的进行去除。

**常用处理：**
  1. 通过箱线图（或 3-Sigma）分析删除异常值；
    就是下面代码的中的示例，采用箱线图进行观察异常值。

 2. BOX-COX 转换（处理有偏分布）；
      对于一些非正态的分布，及偏值较大或者较小的时候，使用Box-Cox转换将数据转换为正态。具体的数学原理就不解释了，[详情了解](https://zhuanlan.zhihu.com/p/53288624)。
具体使用：
 ```
from scipy.special import boxcox1p
all_data[feat] = boxcox1p(all_data[feat], lam)
```
  3. 长尾截断；
  长尾截断主要也是分布不符合正态分布，而是类似于“长尾”。例如房价中，低价占大部分，豪宅属于小部分。应对这种数据分布，一般可以通过神奇的log化处理转化类近似正态分布。

对于### 箱盒图共有两个用途，分别 1. 直观地识别数据中异常值（离群点）；
2. 直观地判断数据离散分布情况，了解数据分布状态。


代码如下：
```
def outer_proc(data,column,scale=3):
    """
    :param data: 接收 pandas 数据格式
    :param column: pandas 列名
    :param scale: 尺度
    :return:
    """
    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值,
        取scale倍的四分之一到四分之三大小的范围
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)
    data_n = data.copy()
    data_series = data_n[column]
    rule , value = box_plot_outliers(data_series,scale)
#   找出异常值的索引   删出去异常值   异常偏的值
    index = np.arange(data_series.shape[0])[rule[0]|rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
#     对于删除后 从新整理一下
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is: {}".format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
#     处理前分布状态
    sns.boxplot(y=data[column], data=data, palette="Set1", ax=ax[0])
#     处理后分布状态
    sns.boxplot(y=data_n[column], data=data_n, palette="Set1", ax=ax[1])
    return data_n
```

**问题：**
> 在处理异常值的时候一般去点什么范围的值？
> 队友的回答 ：
>![image.png](https://upload-images.jianshu.io/upload_images/3426235-ab4ef55fae2d6e6c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## 特征归一化和标准化

### 数据标准化
数据标准化，我的理解是将数据的分布进行线性变化，简单的说就是将数据的转化成特定的分布状态。数据标准化的方法有很多种，常用的有“最小—最大标准化”、“Z-score标准化”和“取对数模式”等。
 
   -  Min-max 标准化：
    Min-max标准化方法是对原始数据进行线性变换。设minA和maxA分别为属性A的最小值和最大值，将A的一个原始值x通过min-max标准化映射成在区间[0,1]中的值x'，其公式为：
      >新数据=（原数据-极小值）/（极大值-极小值）
      >用svm对数据进行训练前一般采用此方法对数据进行标准化。

   - Z-score 标准化:
     这种方法基于原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。将A的原始值x使用Z-score标准化到x'。 Z-score标准化方法适用于属性A的最大值和最小值未知的情况，或有超出取值范围的离群数据的情况。
      >新数据=（原数据-均值）/标准差

  -  取对数模式:
      对于一些长尾数据进行取对数处理，可以使得分布更接近正态分布。

       >  对数Logistic模式：新数据=1/（1+e^(-原数据)）
       > 模糊量化模式：新数据=1/2+1/2*sin[ pi /（极大值-极小值）*（原数据 -（极大值-极小值）/2）]

### 数据归一化
数据归一化指的是将数据缩小到一点的范围，但是不改变数据的分布情况，目的是为了更好的计算，防止太大的数据，计算复杂。 为了数据处理方便提出来的，把数据映射到0～1(或-1~1)范围之内处理，更加便捷快速，应该归到数字信号处理范畴之内。

至于具体的方法这里存在着争议。

因为标准化和归一化都属于四种Feature scaling(特征缩放),这四种分别是：
> 1. Rescaling (min-max normalization) 有时简称normalization(有点坑)
![image.png](https://upload-images.jianshu.io/upload_images/3426235-7fb9780f3bd0fdce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
> 2.Mean normalization 
![image.png](https://upload-images.jianshu.io/upload_images/3426235-e2bc3b9f4e56aaa2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
>  3.Standardization(Z-score normalization)
![image.png](https://upload-images.jianshu.io/upload_images/3426235-6cc9abf90ce69a90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
>  4.Scaling to unit length
![image.png](https://upload-images.jianshu.io/upload_images/3426235-3c6d31b77fedc0b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为什么数据要进行归一化呢，因为数据做归一化的好处 1.加快了梯度下降求最优解的速度和有可能提高精度。2.提高精度，这在涉及到一些距离计算的算法时效果显著，比如算法要计算欧氏距离，所以归一化很有必要，他可以让各个特征对结果做出的贡献相同。

**问题：**
> * 什么模型需要归一化？
>概率模型不需要归一化，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率。像svm、线性回归之类的最优化问题就需要归一化。决策树属于前者。归一化也是提升算法应用能力的必备能力之一。

## 数据分桶：
数据分桶主要是对于连续值的一个处理，因为有些模型需要可能不直接处理连续值，那么我们可以将数据进行分桶，是的连续值变成分类值，可以进一步给模型学习。或者在存在一些缺失值的时候，用分桶的方法，这时候我们的缺失值也进桶了，

为什么要做数据分桶呢，原因有很多?
1. 离散后稀疏向量内积乘法运算速度更快，计算结果也方便存储，容易扩展；
2. 离散后的特征对异常值更具鲁棒性，如 age>30 为 1 否则为 0，对于年龄为 200 的也不会对模型造成很大的干扰；
3. LR 属于广义线性模型，表达能力有限，经过离散化后，每个变量有单独的权重，这相当于引入了非线性，能够提升模型的表达能力，加大拟合；
4. 离散后特征可以进行特征交叉，提升表达能力，由 M+N 个变量编程 M*N 个变量，进一步引入非线形，提升了表达能力；
5. 特征离散后模型更稳定，如用户年龄区间，不会因为用户年龄长了一岁就变化
 
等频分桶；
区间的边界值要经过选择,使得每个区间包含大致相等的实例数量。比如说 N=10 ,每个区间应该包含大约10%的实例
 
等距分桶；
从最小值到最大值之间,均分为 N 等份。 如果 A,B 为最小最大值, 则每个区间的长度为 W=(B?A)/N , 则区间边界值为A+W,A+2W,….A+(N?1)W 。这里只考虑边界，每个等份的实例数量可能不等。

 Best-KS 分桶、卡方分桶：比较复杂 [可参考](https://blog.csdn.net/hxcaifly/article/details/80203663)

```

bin = [i*10 for i in range(31)]
data['power_bin'] = pd.cut(data['power'], bin, labels=False)
data[['power_bin', 'power']].head()
```

### 特征筛选
由于有的时候数据有很多的特征，维度过大，无法将所有的特征进行学习，同时有些特征对于模型的学习预测并不是很少，甚至会是噪音。所以我们要挑选出一些好的特征进行用。当然最好的特征筛选就是了解行业背景，这是最有效的办法。如果这个特征集合有时候也可能很大，在尝试降维之前，我们有必要用特征工程的方法去选择出较重要的特征结合，这些方法不会用到领域知识，而仅仅是统计学的方法。

　　　　最简单的方法就是方差筛选。方差越大的特征，那么我们可以认为它是比较有用的。如果方差较小，比如小于1，那么这个特征可能对我们的算法作用没有那么大。最极端的，如果某个特征方差为0，即所有的样本该特征的取值都是一样的，那么它对我们的模型训练没有任何作用，可以直接舍弃。在实际应用中，我们会指定一个方差的阈值，当方差小于这个阈值的特征会被我们筛掉。sklearn中的VarianceThreshold类可以很方便的完成这个工作。


**过滤式（filter）**：先对数据进行特征选择，然后在训练学习器，常见的方法有
 - 方差选择发：就是上面说的，计算特征的方差，把方差小于阈值的特征筛掉。

 - 相关系数法：指的是通过热力图的方式，计算特征和标签之间的相关系数，小于某个阈值的特征筛掉。
```
# 相关性分析
print(data['power'].corr(data['price'], method='spearman'))
print(data['kilometer'].corr(data['price'], method='spearman'))
print(data['brand_amount'].corr(data['price'], method='spearman'))
print(data['brand_price_average'].corr(data['price'], method='spearman'))
print(data['brand_price_max'].corr(data['price'], method='spearman'))
print(data['brand_price_median'].corr(data['price'], method='spearman'))


#画图法

data_numeric = data[['power', 'kilometer', 'brand_amount', 'brand_price_average', 
                     'brand_price_max', 'brand_price_median']]
correlation = data_numeric.corr()

f , ax = plt.subplots(figsize = (7, 7))
plt.title('Correlation of Numeric Features with Price',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)
```

 - 卡方检验法 ：卡方检验可以检验某个特征分布和输出值分布之间的相关性。个人觉得它比比粗暴的方差法好用。如果大家对卡方检验不熟悉，可以参看这篇[卡方检验原理及应用](https://segmentfault.com/a/1190000003719712 )，这里就不展开了。在sklearn中，可以使用chi2这个类来做卡方检验得到所有特征的卡方值与显著性水平P临界值，我们可以给定卡方值阈值， 选择卡方值较大的部分特征。

 - 互信息法；即从信息熵的角度分析各个特征和输出值之间的关系评分。在[决策树](http://www.cnblogs.com/pinard/p/6050306.html )算法中我们讲到过互信息（信息增益）。互信息值越大，说明该特征和输出值之间的相关性越大，越需要保留。在sklearn中，可以使用mutual_info_classif(分类)和mutual_info_regression(回归)来计算各个输入特征和输出值之间的互信息。

[这篇里面有具体的代码，可以参考！](https://www.cnblogs.com/stevenlk/p/6543628.html)

**包裹式（wrapper）**：直接把最终将要使用的学习器的性能作为特征子集的评价准则，常见方法有 LVM（Las Vegas Wrapper） ；

   最常用的包装法是递归消除特征法(recursive feature elimination,以下简称RFE)。递归消除特征法使用一个机器学习模型来进行多轮训练，每轮训练后，消除若干权值系数的对应的特征，再基于新的特征集进行下一轮训练。在sklearn中，可以使用RFE函数来选择特征。
[LVM的实战案例](https://blog.csdn.net/FontThrone/article/details/79004874)

包裹式好还包括SFS和SBS算法对特征进行过滤，不过不详细展开了，直接附上优秀的[大佬博客](https://www.omegaxyz.com/2018/04/03/sfsandsbs/)、[博客2](https://blog.csdn.net/FontThrone/article/details/79064930)

```
# k_feature 太大会很难跑，没服务器，所以提前 interrupt 了
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
           k_features=10,
           forward=True,
           floating=False,
           scoring = 'r2',
           cv = 0)
x = data.drop(['price'], axis=1)
x = x.fillna(0)
y = data['price']
sfs.fit(x, y)
sfs.k_feature_names_ 
```

**嵌入式（embedding）**：结合过滤式和包裹式，学习器训练过程中自动进行了特征选择，常见的有 lasso 回归；

嵌入法也是用机器学习的方法来选择特征，但是它和RFE的区别是它不是通过不停的筛掉特征来进行训练，而是使用的都是特征全集。在sklearn中，使用SelectFromModel函数来选择特征。

　　　　最常用的是使用L1正则化和L2正则化来选择特征。我们知道正则化惩罚项越大，那么模型的系数就会越小。当正则化惩罚项大到一定的程度的时候，部分特征系数会变成0，当正则化惩罚项继续增大到一定程度时，所有的特征系数都会趋于0\. 但是我们会发现一部分特征系数会更容易先变成0，这部分系数就是可以筛掉的。也就是说，我们选择特征系数较大的特征。常用的L1正则化和L2正则化来选择特征的基学习器是逻辑回归。

　　　　此外也可以使用决策树或者GBDT。那么是不是所有的机器学习方法都可以作为嵌入法的基学习器呢？也不是，一般来说，可以得到特征系数coef或者可以得到特征重要度(feature importances)的算法才可以做为嵌入法的基学习器。

### 特征构造：
- 构造统计量特征，报告计数、求和、比例、标准差等；
```
Train_gb = Train_data.groupby("brand")
all_info = {}
for kind, kind_data in Train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
data = data.merge(brand_fe, how='left', on='brand')
```

 - 时间特征，包括相对时间和绝对时间，节假日，双休日等；
```
# 使用时间：data['creatDate'] - data['regDate']，反应汽车使用时间，一般来说价格与使用时间成反比
# 不过要注意，数据里有时间出错的格式，所以我们需要 errors='coerce'
data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - 
                            pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
```
 - 地理信息，包括分箱，分布编码等方法；
 ```
# 从邮编中提取城市信息，相当于加入了先验知识
data['city'] = data['regionCode'].apply(lambda x : str(x)[:-3])
data = data
```
非线性变换，包括 log/ 平方/ 根号等；

```

data[numeric_features] = np.log(data[numeric_features] + 1)

data[numeric_features] = np.squre(data[numeric_features] + 1)

data[numeric_features] = np.sqrt(data[numeric_features] + 1)
```

### 降维
PCA/ LDA/ ICA；之后打算专门去详细了解一下PCA进行降维的问题。
特征选择也是一种降维。












