function [Y] = BinSVMClassify(X, W, b)

Y = sign(W'*X+b);
