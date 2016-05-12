%
% Create nearest-neighbor graph
%
% Input:
%  X: Input data 
%  k: The number of nearest-neighbor
%  sigma: Heauristics for setting Gaussian width 'median' or 'local-scaling'
%
function [W sigma] = generate_nngraph(X, k, sigma) 

D = squareform(pdist(X').^2);
[sort_D sort_idx] = sort(D);
n = size(X,2);

%% Diagonal entries of D must be ranked the first.
if any(sort_idx(1,:)' ~= (1:n)')
  temp_idx = find(sort_idx(1,:) ~= (1:n));
  [I J] = find(sort_idx(:,temp_idx) == (ones(n,1)*temp_idx));
  if length(I) ~= length(temp_idx)
    error('');
  end  
  for i = 1:length(I)
    temp = sort_idx(I(i),temp_idx(i));
    sort_idx(I(i),temp_idx(i)) = sort_idx(1,temp_idx(i));
    sort_idx(1,temp_idx(i)) = temp;
  end
end

knn_idx = sort_idx(2:k+1,:);
kD = sort_D(2:k+1,:);

W = zeros(n);
if strcmp(sigma,'median') 
  sigma = mean(sqrt(kD(:)));
  if sigma == 0
    sigma = 1;
  end
  for i = 1:n
    W(i,knn_idx(:,i)) = exp(-kD(:,i)./(2*sigma^2));
  end
elseif strcmp(sigma, 'local-scaling') 
  if k < 7
    sigma = sqrt(kD(end,:));
  else
    sigma = sqrt(kD(7,:));
  end
  sigma(sigma == 0) = 1;
  for i = 1:n
    W(i,knn_idx(:,i)) = exp(-kD(:,i)./(sigma(i)*sigma(knn_idx(:,i))'));
  end
else
  error('Unknown option for sigma');
end

W = max(W, W');
