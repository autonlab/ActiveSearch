% Test script for AEW
% (Results depend on random seeds)

clear;

noise_level = 1; % 1 or 2

% Parameters for AEW
param.k = 10; % The number of neighbors in kNN graph
param.sigma = 'median'; % Kernel parameter heuristics 'median' or 'local-scaling'
param.max_iter = 100;

% --------------------------------------------------

[X Y] = generate_syndata(noise_level);
[lb_idx] = select_labeled_nodes(Y,10);
trY = zeros(size(Y));
trY(lb_idx,:) = Y(lb_idx,:);

fprintf('Optimizing edge weights by AEW\n');
[W W0] = AEW(X,param);

fprintf('Estimating labels by harmonic Gaussian model ... ');
L = diag(sum(W)) - W;
F = HGF(L,trY);
L0 = diag(sum(W0)) - W0;
F0 = HGF(L0,trY);
fprintf('done\n');

err_rate = hamming_loss(Y,F) / (size(Y,1) - length(lb_idx));
err_rate0 = hamming_loss(Y,F0) / (size(Y,1) - length(lb_idx));

fprintf('[REPORT]\n');
fprintf('The number of classes = %d\n', size(Y,2));
fprintf('The number of nodes = %d\n', size(Y,1));
fprintf('The number of labeled nodes = %d\n', length(lb_idx));
fprintf('The number of neighbors in kNN graph = %d\n', param.k);
fprintf('The initial kernel parameter heuristics = %s\n', param.sigma);
fprintf('Predction error rate with the inital graph  = %f\n', err_rate0);
fprintf('Predction error rate with the optimized graph = %f\n', err_rate);

