clc; clear; close all;

n = 1000;
r = 30;
nt = 600;
rcross = 0;

d = 100;
num_eval = 200;

init_pt = 1

low = 0;
high = 5/n;
low_c = 0;
high_c = 1/(n^2);

Xt_t = unifrnd(ones(r,nt)*low, ones(r,nt)*high);
Xt_n = unifrnd(ones(r-rcross,nt)*low_c, ones(r-rcross,nt)*high_c);
Xn_n = unifrnd(ones(r,n-nt)*low, ones(r,n-nt)*high);
Xn_t = unifrnd(ones(r-rcross,n- nt)*low_c, ones(r-rcross,n-nt)*high_c);

Xfull = [Xt_t, Xn_t; Xt_n, Xn_n];
Yfull = [ones(nt,1); zeros(n-nt,1)];

A = Xfull'*Xfull;

[Xe, b, w, deg] = glap_eigenmap_3(A,d);

options.start_point = init_pt;
options.n_conncomp = b;
options.num_evaluations = num_eval;
options.prior_prob = sum(Yfull)/length(Yfull);

% [f,hits,selected] = lreg_AS% _main_3(Xe, deg, d, 0.0, Yfull, options);
X = Xe;
labels = Yfull;
alpha = 0.0;
dim = d;
%function [f, hits,selected] = lreg_AS_main_3(X,deg,dim,alpha,labels,options)
%%% [hits,selected] = lreg_AS_main(X,deg,dim,alpha,labels,options) 
%%% Input: 
%%% X: n-by-d matrix. Each row is the feature vector of a node computed by Eigenmap.
%%% deg: n-by-1 vector. Each entry is the sum of pairwise similarity values between a node and all other nodes.
%%% dim: a positive integer indicating the number of leading dimensions in X to use
%%% alpha: a positive real number used in the calculation of the selection score
%%% labels: n-by-1 vector. True labels of data points. 1: target, 0: non-target.
%%% options: a structure specifying values for the following algorithmic options:
%%% 	num_evaluations: number of points we want to investigate (default: 5000)
%%% 	randomSeed: seed of random number generator (default: the current time)
%%% 	log_prefix: prefix of log file name (default: current time string)
%%% 	n_conncomp: number of connected components in the similarity matrix (default 1)
%%% 	omega0: weight assigned to regularization on unlabeled points (default: 1/(#data points))
%%% 	prior_prob: prior target probability (default: 0.05)
%%% 	eta: jump probability (default: 0.5)
%%%
%%% Output:
%%% hits: a vector of cumulative counts of discovered target points
%%% selected: a vector of indices of points selected by the algorithm 

X = X(:,1:dim); % Use the first dim dimensions
n = size(X,1);
sqd = sqrt(deg);
yp = labels .* sqd;

%%% Default values for options
num_evaluations = 5000; %number of points we want to investigate
randomSeed = now; %seed of random number generator, set to the current time
log_prefix = datestr(now,30); %prefix of log file name
b = 1; %number of connected components in the similarity matrix
omega0 = 1/n; %ht assigned to regularization on unlabeled points
pai = 0.5; %prior target probability
eta = 0.5; %jump probability

%%% Set values for options according to input
if(isfield(options,'num_evaluations'))
	num_evaluations = options.num_evaluations;
end
if(isfield(options,'randomSeed'))
	randomSeed = options.randomSeed;
end
if(isfield(options,'log_prefix'))
	log_prefix = options.log_prefix;
end
if(isfield(options,'n_conncomp'))
	b = options.n_conncomp;
end
if(isfield(options,'omega0'))
	omega0 = options.omega0;
end
if(isfield(options,'prior_prob'))
	pai = options.prior_prob;
end
if(isfield(options,'eta'))
	eta = options.eta;
end

lamb = (1-eta)/eta;
r = lamb*omega0;
c = 1/(1-r);
num_initial = 1; % For now we always initialize with 1 target point.
% stream = RandStream('mt19937ar', 'Seed', randomSeed);
% RandStream.setGlobalStream(stream);
% fid = fopen([log_prefix '_seed' num2str(randomSeed) '_alpha' num2str(alpha) '_omega0-' num2str(omega0) '_d' num2str(dim) '.txt'], 'w');

d = size(X,2);
Xp = bsxfun(@times, X, sqd);

%%% Randomly pick 1 target point as the first point
in_train = false(n,1);
f = find(labels==1); randp = randperm(length(f));
start_point = f(randp(1:num_initial));

%%% If user specifies a start point, use it
if(isfield(options,'start_point'))
	start_point = options.start_point;
	disp('Set use-specified start point.')
	if(labels(start_point) ~= 1)
		% printf(2,'Warning: user-specified start point %d is not positive. Change it to positive. \n', start_point);
		labels(start_point) = 1;
		yp(start_point) = labels(start_point) * sqd(start_point);
	end
end

in_train(start_point) = true;
best_ind = start_point;

% fprintf(fid,'start point: %d\n',best_ind);
hits = zeros(num_evaluations+1,1);
selected = zeros(num_evaluations+1,1);
hits(1) = 1;
selected(1) = best_ind;


%%% Initialization of varaibles needed by the algorithm
disp('Initializing variables..');
disp('Constructing C..');
C = r*(Xp'*Xp) + (1-r)*(Xp(best_ind,:)'*Xp(best_ind,:)) + lamb*diag([zeros(b,1); ones(d-b,1)]);
disp('Inverting C..');
C = inv(C);

disp('Computing h and f..');
h = sum((Xp*C).*Xp,2);
% f = r*Xp'*sqd*pai
% f = yp(best_ind)-r*sqd(best_ind)*pai
% f = Xp(best_ind,:)'*(yp(best_ind)-r*sqd(best_ind)*pai)
% f = C * (r*Xp'*sqd*pai + Xp(best_ind,:)'*(yp(best_ind)-r*sqd(best_ind)*pai))
% X = X
f = X * (C * (r*Xp'*sqd*pai + Xp(best_ind,:)'*(yp(best_ind)-r*sqd(best_ind)*pai)));
% f'
disp('Entering main loop..');
%%% Main loop
%format bank
for i = 1:num_evaluations
	tic;

	%%% Calculating change
	test_ind = ~in_train; 
	change = ((((test_ind'*X)*C)*Xp')' - (h./sqd)) .* sqd .* ((1-r*pai)*c-f) ./ (c+h);

	f_bnd = min(max(f(test_ind),0),1);
 
	%%% Calculating selection criteria
	score = f_bnd + alpha*f_bnd.*max(change(test_ind),0);

	%%% select best index
	[best_score best_ind] = max(score);
	best_f = f_bnd(best_ind);
	test_ind = find(test_ind);
	best_ind = test_ind(best_ind);
	best_change = max(change(best_ind),0);
	in_train(best_ind) = true;
    
	if((nnz(labels(in_train)==1))==nnz(labels==1))
		elapsed = toc;
		% fprintf(fid, '%d %d %f\n', i, nnz(labels(in_train)==1), elapsed);
		break;
	end

	%%% Updating parameters
	%keyboard;
	%yp(best_ind) = sqd(best_ind) * input(['Requesting label for e-mail '  num2str(best_ind)  ':']);
	CXp = C * Xp(best_ind,:)';
	f = f + X * ( (CXp*((yp(best_ind)-r*sqd(best_ind)*pai)*c - sqd(best_ind)*f(best_ind)) / (c+h(best_ind))) );
	%f = f + X * CXp * (yp_new(i) - yp(i));
	C = C - (CXp*CXp')/(c+h(best_ind));
	h = h - (Xp*CXp).^2 / (c+h(best_ind));
	elapsed = toc;

	selected(i+1) = best_ind;
	hits(i+1) = nnz(labels(in_train) == 1);
	if(mod(i,1)==0 || i==1)
	  disp([num2str(i) ', #' num2str(selected(i+1)) ', E[u]=' num2str(best_score) ', best f = ' num2str(best_f) ', best change = ' num2str(best_change) ...
	        ', hits ' num2str(hits(i+1)) '/' num2str(i+1)...
	        ', took: ' num2str(elapsed) 's']);
	end
	% fprintf(fid, '%d %d %f %d\n', i, hits(i+1), elapsed, selected(i+1));
end

% % fclose(fid);

save -6 M.mat;