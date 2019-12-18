clc,clear;
close all;


iris = load('iris.mat');   %使用鸢尾花数据集  共三类
data = iris.Iris';
sets = data(: , 2:5);
label = data(: , 1);
lab_max = max(label);
lab_min = min(label);
label = (label-lab_min) / (lab_max-lab_min);

N = 120;   %训练样本数量
M = 150;   %样本总数
train_sets = [sets(1:40 , :) ; sets(51:90 , :) ; sets(101:140 , :)];    %训练样本90个
test_sets = [sets(41:50  , :) ; sets(91:100 , :) ; sets(141:150 , :)];%测试样本60个
train_label = [label(1:40 , :) ; label(51:90 , :) ; label(101:140 , :)];    
test_label = [label(41:50  , :) ; label(91:100 , :) ; label(141:150 , :)];



% sets  = load('testSet.txt');
% [m n] = size(sets);
% N = 100 ;                               %训练集数量
% sets(sets(: , 3) == -1 , 3) = 0;  %将标签中的-1替换为0
% train_sets = sets(1:N , 1:2);     %训练样本数据
% train_label = sets(1:N , 3);
% test_sets = sets(N+1:m , 1:2); %测试样本数据
% test_label = sets(N+1:m , 3);

[mm nn] = size(train_sets);
d = nn;   %训练样本维数
q = 20;     %隐层神经元数量   数量越多 模型越复杂

V =   zeros(d , q);   %输入层到隐层的连接权矩阵  每一列为一个权向量 共q个权向量
W  = zeros(q , 1);   %连接隐层的输出的权向量   只有一个输出
v0 = rand(q , 1);    %隐层阈值
w0 = rand(1);        %输出层阈值

V =  rand(d  , q);    %将权向量随机的初始化为比较小的数值
W = rand(q , 1);

g = 0;  %输出梯度项
e = zeros(1,q);  %隐层梯度项

l = 0.03;  %学习率
b = zeros(q , 1);  %隐层输出
y = zeros(1 , mm);    %估计输出
d_gama = zeros(q , 1);  %误差在隐层阈值上的导数
d_V = zeros(d , q);      %误差在隐层权向量上的导数
count = 0;
K  = 1000;                    %迭代次数
E = zeros(1 , K);
test_acc = zeros(1,K);
train_acc = zeros(1,K);

T1 = 1/3;            %决策阈值
T2 = 2/3;
res = zeros(1 , M-N);
b2 = zeros(1 , q);   %隐层输出

while count < K    %训练次数超过退出训练
    count = count+1;   %训练次数
    for i = 1:mm   %遍历每个训练样本
        for h = 1:q
            b(h) = Sigmoid( train_sets(i , :)*V(: , h) - v0(h) );  %隐层输出
        end
        y(i) =Sigmoid( b' * W  - w0);  %神经网络输出
        g = y(i)*(1-y(i))*(train_label(i) - y(i));   %输出层神经元梯度项
        for h = 1:q
            e(h) = b(h)*(1-b(h))*W(h)*g;        %隐层神经元梯度项
        end
        d_W = l*g*b;                        %误差在输出层连接权上的梯度
        d_theta = -l*g;
        for h = 1:q
            d_V(: , h) = l * e(h) * train_sets(i)' ; %误差在隐层连接权方向的梯度
            d_gama(h) = -l * e(h);
        end
        W = W+d_W;      %更新所有的权连接跟权阈值
        w0 = w0+d_theta;
        V = V+d_V;
        v0 = v0+d_gama;
        
    end
    for i = 1:mm   %遍历每个训练样本
        for h = 1:q
            b(h) = Sigmoid( train_sets(i , :)*V(: , h) - v0(h) );  %隐层输出
        end
        y(i) =Sigmoid( b' * W  - w0);  %神经网络输出
    end
    E(count)  = norm(train_label - y');
    if E(count)<0.8
        break; %累计误差小于E则退出迭代
    end
%     l  = E(count)^2 / 300;   %学习率随着累计误差减小而减小  可以一定程度提高收敛速度
    

% ------------------------验证正确率----------------------%
for i = 1:M-N
    for h = 1:q  %q个隐层神经元
        b2(h) = Sigmoid(test_sets(i , :) * V(: ,  h) - v0(h));
    end
    res(i) = Sigmoid(b2 * W - w0);
end
% res(res>  T2) = 1;
% res((res<= T2) & (res>=T1)) = 0.5;
% res(res < T1) = 0;
% 
% y(y>  T2) = 1;
% y((y<= T2) & (y>=T1)) = 0.5;
% y(y < T1) = 0;
% err = sum(res ~= test_label');
% err_train = sum(y ~= train_label');
% test_acc(count) = (M-N-err) / (M-N);
% train_acc(count) = (N-err_train) / N;
fprintf('第 %d 次训练\n' , count);
% disp('测试正确率')
% disp(test_acc(count));
% disp('训练正确率')
% disp(train_acc(count));
disp(E(count));
 
end

subplot(131); plot(E);   %误差平方和趋势
title('误差平方和趋势');
subplot(132); plot(test_acc);
title('test acc');
subplot(133); plot(train_acc);
title('train acc');


    


