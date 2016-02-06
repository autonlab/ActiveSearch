clc; clear all; close all;

paths = genpath('/usr0/home/sibiv/opt/lapsvmp_v02');
path(path, paths);

load lapsvm_test.mat
Y = Y';
Yt = Yt';

options=make_options('gamma_I',1,'gamma_A',1e-5,'NN',6,'KernelParam',0.35);
options.Verbose=1;
options.UseBias=1;
options.UseHinge=1;
options.LaplacianNormalize=0;
options.NewtonLineSearch=0;

% creating the 'data' structure
data.X=X;
data.Y=Yt;

fprintf('Computing Gram matrix and Laplacian...\n\n');
data.K=calckernel(options,X,X);
data.L = laplacian(options,X);

% % training the classifier
% fprintf('Training LapSVM in the primal with Newton''s method...\n');
% classifier1=lapsvmp(options,data);
% 
% % computing error rate
% fprintf('It took %f seconds.\n',classifier1.traintime);
% out1=sign(data.K(:,classifier1.svs)*classifier1.alpha+classifier1.b);
% er1=100*(length(data.Y)-nnz(out1==Y))/length(data.Y);
% fprintf('Error rate=%.1f\n\n',er1);

% training the classifier
fprintf('Training LapSVM in the primal with early stopped PCG...\n');
options.Cg=1; % PCG
options.MaxIter=1000; % upper bound
options.CgStopType=1; % 'stability' early stop
options.CgStopParam=0.015; % tolerance: 1.5%
options.CgStopIter=3; % check stability every 3 iterations

% dbstop in lapsvmp at 496
classifier2=lapsvmp(options,data);
fprintf('It took %f seconds.\n',classifier2.traintime);

% computing error rate
out2=sign(data.K(:,classifier2.svs)*classifier2.alpha+classifier2.b);
er2=100*(length(data.Y)-nnz(out2==Y))/length(data.Y);
fprintf('Error rate=%.1f\n\n',er2);
