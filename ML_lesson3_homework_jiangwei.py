import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#读取数据
x = pd.read_csv('.\data.csv')

#训练和测试集切分函数
def train_test_split(data_set,split_ratio,x_nd,y_nd):
    train_rows = int(data_set.shape[0]*split_ratio)
    test_rows = data_set.shape[0] - train_rows
    x_train = np.array(data_set.iloc[:,0:x_nd].head(train_rows)).reshape(train_rows,x_nd)
    y_train = np.array(data_set.iloc[:, -1*y_nd].head(train_rows)).reshape(train_rows, y_nd)
    x_test = np.array(data_set.iloc[:, 0:x_nd].tail(test_rows)).reshape(test_rows, x_nd)
    y_test = np.array(data_set.iloc[:, -1*y_nd].tail(test_rows)).reshape(test_rows, y_nd)
    return x_train,y_train,x_test,y_test


#定义模型
def model(x,w,b):
    return np.dot(x,w.T) + b


#定义损失函数
def cost(x,y,w,b):
    m = np.shape(x)[0]
    costs = 0.5/m * np.square(np.dot(x,w.T)+b-y).sum()
    return costs


#优化器
def optimize(x,y,w,b):
    m = np.shape(x)[0]
    alpha = 1e-6        # 学习率大于e-5将cost函数将报溢出错误
    y_pred = model(x,w,b)
    d_w = (1.0/m)*(np.dot((y_pred-y).T,x))
    d_b = (1.0/m)*((y_pred-y).sum())
    w = w -alpha*d_w
    b = b -alpha*d_b
    return w,b

#迭代器
def iterate(x,y,w,b,n):
    for i in range(n):
        w,b = optimize(x,y,w,b)
    y_pred = model(x,w,b)
    cost_ = cost(x,y,w,b)
    # print(w,b,cost_)
    plt.scatter(y,y_pred)
    # plt.plot(y,y_pred)
    plt.show()
    return w,b

#训练函数
def train(x,y,n):
    w = np.array([[1,1,1,1]])
    b = 0
    w,b = iterate(x,y,w,b,n)
    return w,b

#验证函数
def validate(w,b,x,y):
    y_pred = model(x,w,b)
    y_mean = y.mean()
    SSR = np.square(y_pred-y_mean).sum()
    SST = np.square(y-y_mean).sum()
    R2 = SSR/SST
    print(R2)

#运行预测模型，返回训练结果
if __name__ == '__main__':
    x_train,y_train,x_test,y_test = train_test_split(x,.4,4,1)
    w, b = train(x_train, y_train, 1000000)
    print("w==> :" + str(w))
    print("b==> :" + str(b))
    print("train result:")
    validate(w, b, x_train, y_train)
    print("*"*60)
    print("test result:")
    validate(w,b,x_test,y_test)