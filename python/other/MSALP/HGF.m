%
% Harmonic Gaussian Field (HGF) model for label propagation
%
% INPUT
%  L: (combinatorial) graph Laplacian (n times n)
%  Y: Label indicator matrix (n times num. of classes)
% OUTPUT
%  F: Estimator matrix for Y
%
function F = HGF(L, Y)

  n = size(Y,1);
  lb_idx = find(sum(Y,2));
  ul_idx = setdiff(1:n,lb_idx);

  F_ul = - L(ul_idx,ul_idx) \ (L(ul_idx,lb_idx)*Y(lb_idx,:));
  F(lb_idx,:) = Y(lb_idx,:);
  F(ul_idx,:) = F_ul;