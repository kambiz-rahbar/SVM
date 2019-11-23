clc;
clear;
close all;

X = 2*rand(2,15)-1;
T = -ones(size(X,2),1)';
X = [X X+3];
T = [T -T];

% set SVM parameters
C = 5;
MaxIter = 100;
alphaTresholdScale = 10^-5;

% train binary SVM
[W, b, SupVec] = trainBinSVM(X, T, C, alphaTresholdScale, MaxIter);

% test binary SVM
Y = BinSVMClassify(X, W, b);

% check performance
SVM_Performance(T,Y)

% Plot Results
figure(2);
hold on;
plot(X(1,T==-1),X(2,T==-1),'ko','linewidth',2);
plot(X(1,T==1),X(2,T==1),'kx','linewidth',2);

Xmin = min(X(2, :));
Ymin = -1/W(2) * (W(1)*Xmin+b);

Xmax = max(X(2, :));
Ymax = -1/W(2) * (W(1)*Xmax+b);

plot([Xmin Xmax] , [Ymin Ymax],'r','linewidth',2);
plot([Xmin Xmax] , [Ymin Ymax]-1/W(2),'-.g','linewidth',2);
plot([Xmin Xmax] , [Ymin Ymax]+1/W(2),'-.b','linewidth',2);

plot(SupVec(1,:), SupVec(2,:), 'ob','MarkerSize',15,'linewidth',2);

xlabel('X1');
ylabel('X2');
legend('class #1', 'class #2',...
    sprintf('[%.2f %.2f]''*X + (%.2f) = 0',W,b),...
    sprintf('[%.2f %.2f]''*X + (%.2f) = 1',W,b),...
    sprintf('[%.2f %.2f]''*X + (%.2f) = -1',W,b),...
    sprintf('SupVec:(%d)',size(SupVec,2)));


