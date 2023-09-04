# 导入最基本的数据处理工具
import pandas as pd # 导入Pandas数据处理工具包
df_ads = pd.read_csv('微信打赏.csv') # 读入数据
df_ads.head(10) # 显示前几行数据
X = df_ads.drop(['打赏金'],axis=1) # 特征集，Drop掉便签相关字段
y = df_ads.打赏金 # 标签集
X.head() # 显示前几行数据
# 将数据集进行80%（训练集）和20%（验证集）的分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression # 导入线性回归算法模型
model = LinearRegression() # 使用线性回归算法创建模型
model.fit(X_train, y_train) # 用训练集数据，训练机器，拟合函数，确定参数
y_pred = model.predict(X_test) #预测测试集的Y值
df_ads_pred = X_test.copy() #测试集特征数据
df_ads_pred['打赏金额真值'] = y_test #测试集标签真值
df_ads_pred['打赏金额预测值'] = y_pred #测试集标签预测值
df_ads_pred #显示数据