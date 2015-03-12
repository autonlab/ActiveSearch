 function [X,n_conncomp,w] = glap_eigenmap(A,d);

deg = sum(A,2);
%if(exist('type','var') && strcmp(type,'normalized'))
%	L = speye(size(A,1)) - diag(1./sqrt(deg))*A*diag(1./sqrt(deg));
%else
%	L = diag(deg) - A;
%end
L = diag(deg) - A;

if(d > size(A,2))
	fprintf(2,'Error: required dimension larger than size of A!');
	return;
end

disp('Constructing Eigenmaps...');
[X lambda] = eig(full(L));
lambda = diag(lambda);
[lambda, perm] = sort(lambda, 'ascend');

th_zero = 1/size(A,1)/1e3;
b = sum(lambda < th_zero);

%keyboard;

w = 1./sqrt(lambda((b+1):d));
X = [X(:,perm(1:b)) bsxfun(@times, X(:,perm((b+1):d)), reshape(w,1,length(w)))];
n_conncomp = b;