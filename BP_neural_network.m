clc,clear;
close all;


iris = load('iris.mat');   %ʹ���β�����ݼ�  ������
data = iris.Iris';
sets = data(: , 2:5);
label = data(: , 1);
lab_max = max(label);
lab_min = min(label);
label = (label-lab_min) / (lab_max-lab_min);

N = 120;   %ѵ����������
M = 150;   %��������
train_sets = [sets(1:40 , :) ; sets(51:90 , :) ; sets(101:140 , :)];    %ѵ������90��
test_sets = [sets(41:50  , :) ; sets(91:100 , :) ; sets(141:150 , :)];%��������60��
train_label = [label(1:40 , :) ; label(51:90 , :) ; label(101:140 , :)];    
test_label = [label(41:50  , :) ; label(91:100 , :) ; label(141:150 , :)];



% sets  = load('testSet.txt');
% [m n] = size(sets);
% N = 100 ;                               %ѵ��������
% sets(sets(: , 3) == -1 , 3) = 0;  %����ǩ�е�-1�滻Ϊ0
% train_sets = sets(1:N , 1:2);     %ѵ����������
% train_label = sets(1:N , 3);
% test_sets = sets(N+1:m , 1:2); %������������
% test_label = sets(N+1:m , 3);

[mm nn] = size(train_sets);
d = nn;   %ѵ������ά��
q = 20;     %������Ԫ����   ����Խ�� ģ��Խ����

V =   zeros(d , q);   %����㵽���������Ȩ����  ÿһ��Ϊһ��Ȩ���� ��q��Ȩ����
W  = zeros(q , 1);   %��������������Ȩ����   ֻ��һ�����
v0 = rand(q , 1);    %������ֵ
w0 = rand(1);        %�������ֵ

V =  rand(d  , q);    %��Ȩ��������ĳ�ʼ��Ϊ�Ƚ�С����ֵ
W = rand(q , 1);

g = 0;  %����ݶ���
e = zeros(1,q);  %�����ݶ���

l = 0.03;  %ѧϰ��
b = zeros(q , 1);  %�������
y = zeros(1 , mm);    %�������
d_gama = zeros(q , 1);  %�����������ֵ�ϵĵ���
d_V = zeros(d , q);      %���������Ȩ�����ϵĵ���
count = 0;
K  = 1000;                    %��������
E = zeros(1 , K);
test_acc = zeros(1,K);
train_acc = zeros(1,K);

T1 = 1/3;            %������ֵ
T2 = 2/3;
res = zeros(1 , M-N);
b2 = zeros(1 , q);   %�������

while count < K    %ѵ�����������˳�ѵ��
    count = count+1;   %ѵ������
    for i = 1:mm   %����ÿ��ѵ������
        for h = 1:q
            b(h) = Sigmoid( train_sets(i , :)*V(: , h) - v0(h) );  %�������
        end
        y(i) =Sigmoid( b' * W  - w0);  %���������
        g = y(i)*(1-y(i))*(train_label(i) - y(i));   %�������Ԫ�ݶ���
        for h = 1:q
            e(h) = b(h)*(1-b(h))*W(h)*g;        %������Ԫ�ݶ���
        end
        d_W = l*g*b;                        %��������������Ȩ�ϵ��ݶ�
        d_theta = -l*g;
        for h = 1:q
            d_V(: , h) = l * e(h) * train_sets(i)' ; %�������������Ȩ������ݶ�
            d_gama(h) = -l * e(h);
        end
        W = W+d_W;      %�������е�Ȩ���Ӹ�Ȩ��ֵ
        w0 = w0+d_theta;
        V = V+d_V;
        v0 = v0+d_gama;
        
    end
    for i = 1:mm   %����ÿ��ѵ������
        for h = 1:q
            b(h) = Sigmoid( train_sets(i , :)*V(: , h) - v0(h) );  %�������
        end
        y(i) =Sigmoid( b' * W  - w0);  %���������
    end
    E(count)  = norm(train_label - y');
    if E(count)<0.8
        break; %�ۼ����С��E���˳�����
    end
%     l  = E(count)^2 / 300;   %ѧϰ�������ۼ�����С����С  ����һ���̶���������ٶ�
    

% ------------------------��֤��ȷ��----------------------%
for i = 1:M-N
    for h = 1:q  %q��������Ԫ
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
fprintf('�� %d ��ѵ��\n' , count);
% disp('������ȷ��')
% disp(test_acc(count));
% disp('ѵ����ȷ��')
% disp(train_acc(count));
disp(E(count));
 
end

subplot(131); plot(E);   %���ƽ��������
title('���ƽ��������');
subplot(132); plot(test_acc);
title('test acc');
subplot(133); plot(train_acc);
title('train acc');


    


