##��Task3 ��������

----


>day3:������е����������̲��֣�Ҳ���Ƕ�һЩ�������д��������ʺϸ���ģ�͵����ݡ�

### �������̵�Ŀ��
�����������н�һ�����������������ݽ��д���

��ɶ����������̵ķ��������������ݽ���һЩͼ����������ܽᲢ�򿨡�

## ����Ԥ����
 - ��ֵ����������ϴ������total_rooms ����λ�����棬ocean_proximity ��one-hot-encode����תΪ��ֵ�ͣ�one-hot-encode��ֱ�ӱ���Ϊ [0,1,2,3..] ��ȣ������������ƻ�����������塣���� 2 �� 3 ����ֵ��������������Ǹ��Ա�ʾ��NEAR BAY��INLAND�����������ʵ�ʵ����ƹ�ϵ��������������ܣ���

 - �������������ݼ�����Ϊ��ͬdistrict�ﻧ����ͬ������ total_rooms��total_bedrooms��population ��Щ���Կ���û��̫������壬��ÿ���ķ��������������������������ϣ����������ݼ��п���������Щ������

 - ��һ������ֵ��������һ���ص��ǣ���ͬ������ȡֵ��Χ��𼫴󣬶�ģ��ѧϰ�����������ò����������ּ򵥵ķ�������Ӧ������������任��0-1����Ĺ�һ�����߱�׼��̬�ֲ���ǰ�߶��쳣��ǳ����У��������ĳ��ֵ������ֵ��������Զ�����ִ���ᵼ���������ݵ�ֲ����ڼ��У����߶Թ��ڼ��е����ݷֲ����У����ԭ���ݼ������С�����Խ�0����ֵ���׵��¾���ʧ�档

 - ƫ�ȴ����ڴ󲿷���ֵ�������У�ͨ���ֲ���������̬�ֲ������������ڡ���β�������緿���У��ͼ�ռ�󲿷֣���լ����С���֡�Ӧ���������ݷֲ���һ�����ͨ�������log������ת���������̬�ֲ�

## �쳣ֵ����
���ڲ�ͬ�����������ǻ�������ͼ����С����ͼ���ӻ����ݵķֲ���������������������̬�ֲ��Ļ������ǻ����԰���3��׼�򣬽�3�����������ȥ��������ѧϰ��ȥ���ķ����ǽ����ݵ�25%-75%��Χ��չbox_scale�����ڴ˷�Χ����Ľ���ȥ����

**���ô���**
  1. ͨ������ͼ���� 3-Sigma������ɾ���쳣ֵ��
    �������������е�ʾ������������ͼ���й۲��쳣ֵ��

 2. BOX-COX ת����������ƫ�ֲ�����
      ����һЩ����̬�ķֲ�����ƫֵ�ϴ���߽�С��ʱ��ʹ��Box-Coxת��������ת��Ϊ��̬���������ѧԭ��Ͳ������ˣ�[�����˽�](https://zhuanlan.zhihu.com/p/53288624)��
����ʹ�ã�
 ```
from scipy.special import boxcox1p
all_data[feat] = boxcox1p(all_data[feat], lam)
```
  3. ��β�ضϣ�
  ��β�ض���ҪҲ�Ƿֲ���������̬�ֲ������������ڡ���β�������緿���У��ͼ�ռ�󲿷֣���լ����С���֡�Ӧ���������ݷֲ���һ�����ͨ�������log������ת���������̬�ֲ���

����### ���ͼ����������;���ֱ� 1. ֱ�۵�ʶ���������쳣ֵ����Ⱥ�㣩��
2. ֱ�۵��ж�������ɢ�ֲ�������˽����ݷֲ�״̬��


�������£�
```
def outer_proc(data,column,scale=3):
    """
    :param data: ���� pandas ���ݸ�ʽ
    :param column: pandas ����
    :param scale: �߶�
    :return:
    """
    def box_plot_outliers(data_ser, box_scale):
        """
        ��������ͼȥ���쳣ֵ,
        ȡscale�����ķ�֮һ���ķ�֮����С�ķ�Χ
        :param data_ser: ���� pandas.Series ���ݸ�ʽ
        :param box_scale: ����ͼ�߶ȣ�
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
#   �ҳ��쳣ֵ������   ɾ��ȥ�쳣ֵ   �쳣ƫ��ֵ
    index = np.arange(data_series.shape[0])[rule[0]|rule[1]]
    print("Delete number is: {}".format(len(index)))
    data_n = data_n.drop(index)
#     ����ɾ���� ��������һ��
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
#     ����ǰ�ֲ�״̬
    sns.boxplot(y=data[column], data=data, palette="Set1", ax=ax[0])
#     �����ֲ�״̬
    sns.boxplot(y=data_n[column], data=data_n, palette="Set1", ax=ax[1])
    return data_n
```

**���⣺**
> �ڴ����쳣ֵ��ʱ��һ��ȥ��ʲô��Χ��ֵ��
> ���ѵĻش� ��
>![image.png](https://upload-images.jianshu.io/upload_images/3426235-ab4ef55fae2d6e6c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## ������һ���ͱ�׼��

### ���ݱ�׼��
���ݱ�׼�����ҵ�����ǽ����ݵķֲ��������Ա仯���򵥵�˵���ǽ����ݵ�ת�����ض��ķֲ�״̬�����ݱ�׼���ķ����кܶ��֣����õ��С���С������׼��������Z-score��׼�����͡�ȡ����ģʽ���ȡ�
 
   -  Min-max ��׼����
    Min-max��׼�������Ƕ�ԭʼ���ݽ������Ա任����minA��maxA�ֱ�Ϊ����A����Сֵ�����ֵ����A��һ��ԭʼֵxͨ��min-max��׼��ӳ���������[0,1]�е�ֵx'���乫ʽΪ��
      >������=��ԭ����-��Сֵ��/������ֵ-��Сֵ��
      >��svm�����ݽ���ѵ��ǰһ����ô˷��������ݽ��б�׼����

   - Z-score ��׼��:
     ���ַ�������ԭʼ���ݵľ�ֵ��mean���ͱ�׼�standard deviation���������ݵı�׼������A��ԭʼֵxʹ��Z-score��׼����x'�� Z-score��׼����������������A�����ֵ����Сֵδ֪����������г���ȡֵ��Χ����Ⱥ���ݵ������
      >������=��ԭ����-��ֵ��/��׼��

  -  ȡ����ģʽ:
      ����һЩ��β���ݽ���ȡ������������ʹ�÷ֲ����ӽ���̬�ֲ���

       >  ����Logisticģʽ��������=1/��1+e^(-ԭ����)��
       > ģ������ģʽ��������=1/2+1/2*sin[ pi /������ֵ-��Сֵ��*��ԭ���� -������ֵ-��Сֵ��/2��]

### ���ݹ�һ��
���ݹ�һ��ָ���ǽ�������С��һ��ķ�Χ�����ǲ��ı����ݵķֲ������Ŀ����Ϊ�˸��õļ��㣬��ֹ̫������ݣ����㸴�ӡ� Ϊ�����ݴ�����������ģ�������ӳ�䵽0��1(��-1~1)��Χ֮�ڴ������ӱ�ݿ��٣�Ӧ�ù鵽�����źŴ�����֮�ڡ�

���ھ���ķ���������������顣

��Ϊ��׼���͹�һ������������Feature scaling(��������),�����ֱַ��ǣ�
> 1. Rescaling (min-max normalization) ��ʱ���normalization(�е��)
![image.png](https://upload-images.jianshu.io/upload_images/3426235-7fb9780f3bd0fdce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
> 2.Mean normalization 
![image.png](https://upload-images.jianshu.io/upload_images/3426235-e2bc3b9f4e56aaa2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
>  3.Standardization(Z-score normalization)
![image.png](https://upload-images.jianshu.io/upload_images/3426235-6cc9abf90ce69a90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
>  4.Scaling to unit length
![image.png](https://upload-images.jianshu.io/upload_images/3426235-3c6d31b77fedc0b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Ϊʲô����Ҫ���й�һ���أ���Ϊ��������һ���ĺô� 1.�ӿ����ݶ��½������Ž���ٶȺ��п�����߾��ȡ�2.��߾��ȣ������漰��һЩ���������㷨ʱЧ�������������㷨Ҫ����ŷ�Ͼ��룬���Թ�һ�����б�Ҫ���������ø��������Խ�������Ĺ�����ͬ��

**���⣺**
> * ʲôģ����Ҫ��һ����
>����ģ�Ͳ���Ҫ��һ������Ϊ���ǲ����ı�����ֵ�����ǹ��ı����ķֲ��ͱ���֮����������ʡ���svm�����Իع�֮������Ż��������Ҫ��һ��������������ǰ�ߡ���һ��Ҳ�������㷨Ӧ�������ıر�����֮һ��

## ���ݷ�Ͱ��
���ݷ�Ͱ��Ҫ�Ƕ�������ֵ��һ��������Ϊ��Щģ����Ҫ���ܲ�ֱ�Ӵ�������ֵ����ô���ǿ��Խ����ݽ��з�Ͱ���ǵ�����ֵ��ɷ���ֵ�����Խ�һ����ģ��ѧϰ�������ڴ���һЩȱʧֵ��ʱ���÷�Ͱ�ķ�������ʱ�����ǵ�ȱʧֵҲ��Ͱ�ˣ�

ΪʲôҪ�����ݷ�Ͱ�أ�ԭ���кܶ�?
1. ��ɢ��ϡ�������ڻ��˷������ٶȸ��죬������Ҳ����洢��������չ��
2. ��ɢ����������쳣ֵ����³���ԣ��� age>30 Ϊ 1 ����Ϊ 0����������Ϊ 200 ��Ҳ�����ģ����ɺܴ�ĸ��ţ�
3. LR ���ڹ�������ģ�ͣ�����������ޣ�������ɢ����ÿ�������е�����Ȩ�أ����൱�������˷����ԣ��ܹ�����ģ�͵ı���������Ӵ���ϣ�
4. ��ɢ���������Խ����������棬��������������� M+N ��������� M*N ����������һ����������Σ������˱��������
5. ������ɢ��ģ�͸��ȶ������û��������䣬������Ϊ�û����䳤��һ��ͱ仯
 
��Ƶ��Ͱ��
����ı߽�ֵҪ����ѡ��,ʹ��ÿ���������������ȵ�ʵ������������˵ N=10 ,ÿ������Ӧ�ð�����Լ10%��ʵ��
 
�Ⱦ��Ͱ��
����Сֵ�����ֵ֮��,����Ϊ N �ȷݡ� ��� A,B Ϊ��С���ֵ, ��ÿ������ĳ���Ϊ W=(B?A)/N , ������߽�ֵΪA+W,A+2W,��.A+(N?1)W ������ֻ���Ǳ߽磬ÿ���ȷݵ�ʵ���������ܲ��ȡ�

 Best-KS ��Ͱ��������Ͱ���Ƚϸ��� [�ɲο�](https://blog.csdn.net/hxcaifly/article/details/80203663)

```

bin = [i*10 for i in range(31)]
data['power_bin'] = pd.cut(data['power'], bin, labels=False)
data[['power_bin', 'power']].head()
```

### ����ɸѡ
�����е�ʱ�������кܶ��������ά�ȹ����޷������е���������ѧϰ��ͬʱ��Щ��������ģ�͵�ѧϰԤ�Ⲣ���Ǻ��٣�����������������������Ҫ��ѡ��һЩ�õ����������á���Ȼ��õ�����ɸѡ�����˽���ҵ��������������Ч�İ취������������������ʱ��Ҳ���ܴܺ��ڳ��Խ�ά֮ǰ�������б�Ҫ���������̵ķ���ȥѡ�������Ҫ��������ϣ���Щ���������õ�����֪ʶ����������ͳ��ѧ�ķ�����

����������򵥵ķ������Ƿ���ɸѡ������Խ�����������ô���ǿ�����Ϊ���ǱȽ����õġ���������С������С��1����ô����������ܶ����ǵ��㷨����û����ô����˵ģ����ĳ����������Ϊ0�������е�������������ȡֵ����һ���ģ���ô�������ǵ�ģ��ѵ��û���κ����ã�����ֱ����������ʵ��Ӧ���У����ǻ�ָ��һ���������ֵ��������С�������ֵ�������ᱻ����ɸ����sklearn�е�VarianceThreshold����Ժܷ����������������


**����ʽ��filter��**���ȶ����ݽ�������ѡ��Ȼ����ѵ��ѧϰ���������ķ�����
 - ����ѡ�񷢣���������˵�ģ����������ķ���ѷ���С����ֵ������ɸ����

 - ���ϵ������ָ����ͨ������ͼ�ķ�ʽ�����������ͱ�ǩ֮������ϵ����С��ĳ����ֵ������ɸ����
```
# ����Է���
print(data['power'].corr(data['price'], method='spearman'))
print(data['kilometer'].corr(data['price'], method='spearman'))
print(data['brand_amount'].corr(data['price'], method='spearman'))
print(data['brand_price_average'].corr(data['price'], method='spearman'))
print(data['brand_price_max'].corr(data['price'], method='spearman'))
print(data['brand_price_median'].corr(data['price'], method='spearman'))


#��ͼ��

data_numeric = data[['power', 'kilometer', 'brand_amount', 'brand_price_average', 
                     'brand_price_max', 'brand_price_median']]
correlation = data_numeric.corr()

f , ax = plt.subplots(figsize = (7, 7))
plt.title('Correlation of Numeric Features with Price',y=1,size=16)
sns.heatmap(correlation,square = True,  vmax=0.8)
```

 - �������鷨 ������������Լ���ĳ�������ֲ������ֵ�ֲ�֮�������ԡ����˾������ȱȴֱ��ķ�����á������ҶԿ������鲻��Ϥ�����Բο���ƪ[��������ԭ��Ӧ��](https://segmentfault.com/a/1190000003719712 )������Ͳ�չ���ˡ���sklearn�У�����ʹ��chi2�����������������õ����������Ŀ���ֵ��������ˮƽP�ٽ�ֵ�����ǿ��Ը�������ֵ��ֵ�� ѡ�񿨷�ֵ�ϴ�Ĳ���������

 - ����Ϣ����������Ϣ�صĽǶȷ����������������ֵ֮��Ĺ�ϵ���֡���[������](http://www.cnblogs.com/pinard/p/6050306.html )�㷨�����ǽ���������Ϣ����Ϣ���棩������ϢֵԽ��˵�������������ֵ֮��������Խ��Խ��Ҫ��������sklearn�У�����ʹ��mutual_info_classif(����)��mutual_info_regression(�ع�)����������������������ֵ֮��Ļ���Ϣ��

[��ƪ�����о���Ĵ��룬���Բο���](https://www.cnblogs.com/stevenlk/p/6543628.html)

**����ʽ��wrapper��**��ֱ�Ӱ����ս�Ҫʹ�õ�ѧϰ����������Ϊ�����Ӽ�������׼�򣬳��������� LVM��Las Vegas Wrapper�� ��

   ��õİ�װ���ǵݹ�����������(recursive feature elimination,���¼��RFE)���ݹ�����������ʹ��һ������ѧϰģ�������ж���ѵ����ÿ��ѵ������������Ȩֵϵ���Ķ�Ӧ���������ٻ����µ�������������һ��ѵ������sklearn�У�����ʹ��RFE������ѡ��������
[LVM��ʵս����](https://blog.csdn.net/FontThrone/article/details/79004874)

����ʽ�û�����SFS��SBS�㷨���������й��ˣ���������ϸչ���ˣ�ֱ�Ӹ��������[���в���](https://www.omegaxyz.com/2018/04/03/sfsandsbs/)��[����2](https://blog.csdn.net/FontThrone/article/details/79064930)

```
# k_feature ̫�������ܣ�û��������������ǰ interrupt ��
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

**Ƕ��ʽ��embedding��**����Ϲ���ʽ�Ͱ���ʽ��ѧϰ��ѵ���������Զ�����������ѡ�񣬳������� lasso �ع飻

Ƕ�뷨Ҳ���û���ѧϰ�ķ�����ѡ����������������RFE��������������ͨ����ͣ��ɸ������������ѵ��������ʹ�õĶ�������ȫ������sklearn�У�ʹ��SelectFromModel������ѡ��������

����������õ���ʹ��L1���򻯺�L2������ѡ������������֪�����򻯳ͷ���Խ����ôģ�͵�ϵ���ͻ�ԽС�������򻯳ͷ����һ���ĳ̶ȵ�ʱ�򣬲�������ϵ������0�������򻯳ͷ����������һ���̶�ʱ�����е�����ϵ����������0\. �������ǻᷢ��һ��������ϵ����������ȱ��0���ⲿ��ϵ�����ǿ���ɸ���ġ�Ҳ����˵������ѡ������ϵ���ϴ�����������õ�L1���򻯺�L2������ѡ�������Ļ�ѧϰ�����߼��ع顣

������������Ҳ����ʹ�þ���������GBDT����ô�ǲ������еĻ���ѧϰ������������ΪǶ�뷨�Ļ�ѧϰ���أ�Ҳ���ǣ�һ����˵�����Եõ�����ϵ��coef���߿��Եõ�������Ҫ��(feature importances)���㷨�ſ�����ΪǶ�뷨�Ļ�ѧϰ����

### �������죺
- ����ͳ���������������������͡���������׼��ȣ�
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

 - ʱ���������������ʱ��;���ʱ�䣬�ڼ��գ�˫���յȣ�
```
# ʹ��ʱ�䣺data['creatDate'] - data['regDate']����Ӧ����ʹ��ʱ�䣬һ����˵�۸���ʹ��ʱ��ɷ���
# ����Ҫע�⣬��������ʱ�����ĸ�ʽ������������Ҫ errors='coerce'
data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - 
                            pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
```
 - ������Ϣ���������䣬�ֲ�����ȷ�����
 ```
# ���ʱ�����ȡ������Ϣ���൱�ڼ���������֪ʶ
data['city'] = data['regionCode'].apply(lambda x : str(x)[:-3])
data = data
```
�����Ա任������ log/ ƽ��/ ���ŵȣ�

```

data[numeric_features] = np.log(data[numeric_features] + 1)

data[numeric_features] = np.squre(data[numeric_features] + 1)

data[numeric_features] = np.sqrt(data[numeric_features] + 1)
```

### ��ά
PCA/ LDA/ ICA��֮�����ר��ȥ��ϸ�˽�һ��PCA���н�ά�����⡣
����ѡ��Ҳ��һ�ֽ�ά��












