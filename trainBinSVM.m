function [W, b, Sx] = trainBinSVM(X, T, C, alphaTresholdScale, MaxIter)

N = length(T);

XX = zeros(N,N);
for i = 1:N
    for j = 1:N
        XX(i,j) = X(:,i)'*X(:,j);
    end
end
M = repmat(T,N,1).*repmat(T',1,N).*XX;

O = -ones(N,1);

QP_options = optimset('Algorithm', 'interior-point-convex', 'MaxIter', MaxIter);
alpha = quadprog(M,O,[],[],T,0,zeros(N,1),C*ones(N,1),[],QP_options)';
alphaThreshold = mean(abs(alpha)) * alphaTresholdScale;
alpha(abs(alpha) <= alphaThreshold) = 0;

W = sum(alpha.*T.*X, 2);

Sx = X(:, alpha>0 & alpha<C);
Sy = T(:, alpha>0 & alpha<C);

b = mean(Sy - W'*Sx);


