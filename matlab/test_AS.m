clc; clear; close all;

X = eye(10) + diag(ones(9,1),1);
Xfull = blkdiag(X,X);
Yfull = [ones(1,10), zeros(1,10)]';

A = Xfull'*Xfull;
d = 4;

[Xe, b, w, deg] = glap_eigenmap_3(A,d);

options.start_point = 1;
options.n_conncomp = b;
options.num_evaluations = 5;
options.pai = 0.5;

[f,hits,selected] = lreg_AS_main_3(Xe, deg, d, 0.0, Yfull, options);