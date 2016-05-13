function []=Compare(x,y,T,s,Kp,Kn);
N = size(x,1);
one = ones(N,1);

% plot data
subplot(2,2,1);
%scatter(x(T==1),y(T==1)); hold on; scatter(x(T~=1),y(T~=1)); hold off; title('ground truth');
color = T*[0,0,0.9]+(1-T)*[0.9,0,0];
scatter(x,y,[],color);
title('ground truth');

% create adjacency matrix
A=exp( -((x*one'-one*x').^2+(y*one'-one*y').^2)/(2*s*s) );
%A(A<1e-100) = 1e-100;

% cluster
idx_k = kmeans(A,2)==1;
[V,~] = eig(diag(1./sum(A,2))*A);
idx_s = V(:,2)>0;
% figure();
% scatter(x(idx_k),y(idx_k)); hold on; scatter(x(~idx_k),y(~idx_k)); hold off; title('k means');
% figure();
% scatter(x(idx_s),y(idx_s)); hold on; scatter(x(~idx_s),y(~idx_s)); hold off; title('spectral');

% AS
pos_idx = find(T==1); pos_idx=pos_idx(1:Kp); 
neg_idx = find(T~=1); neg_idx=neg_idx(1:Kn);
yhat = one*0.5; yhat(pos_idx) = 1; yhat(neg_idx) = 0;
S = one/N; S([pos_idx;neg_idx]) = 9;
D = diag(sum(A,2)); B = D*diag(S);
f = (D+B-A)\(B*yhat);
wn = (sum(A(:,pos_idx),2)-sum(A(:,neg_idx),2))./(sum(A(:,pos_idx),2)+sum(A(:,neg_idx),2));
wn(isnan(wn)) = 0;

l = zeros(N,1); l([pos_idx;neg_idx])=1;
minmax = @(x,lower,upper) min(max(x,lower),upper);
color1f = @(x) ones(size(x))*[0.8,0.8,1]+minmax((x-min(x(~l)))/(max(x(~l)-min(x(~l)))),0,1)*([0,0,0.9]-[0.8,0.8,1]);
color2f = @(x) ones(size(x))*[0.9,0,0]+minmax((x-min(x(~l)))/(max(x(~l)-min(x(~l)))),0,1)*([1,0.8,0.8]-[0.9,0,0]);
color3f = @(x) ones(size(x))*[0.5,0.5,0.5];
colorf = @(x,tau) color1f(x).*repmat(x>tau,1,3)+color2f(x).*repmat(x<tau,1,3)+color3f(x).*repmat(x==tau,1,3);

subplot(2,2,2);
%scatter(x(wn>0),y(wn>0)); hold on; scatter(x(wn<0),y(wn<0)); scatter(x(wn==0),y(wn==0),[],[0.5,0.5,0.5]);
scatter(x,y,[],colorf(wn,0)); hold on;
scatter(x(pos_idx),y(pos_idx),200,'+','MarkerEdgeColor',[0,0,0],'LineWidth',3); scatter(x(neg_idx),y(neg_idx),200,'x','MarkerEdgeColor',[0,0,0],'LineWidth',3); hold off;
title('weighted neighbors');
subplot(2,2,3);
%scatter(x(f>0.5),y(f>0.5)); hold on; scatter(x(f<0.5),y(f<0.5));  scatter(x(f==0),y(f==0),[],[0.5,0.5,0.5]);
scatter(x,y,[],colorf(f,0.5)); hold on;
scatter(x(pos_idx),y(pos_idx),200,'+','MarkerEdgeColor',[0,0,0],'LineWidth',3); scatter(x(neg_idx),y(neg_idx),200,'x','MarkerEdgeColor',[0,0,0],'LineWidth',3); hold off;
title('active search');

% AS2
%L = eye(2*N)-diag(1./sum(A,2))*A;
L = eye(N)-diag(sqrt(1./sum(A,2)))*A*diag(sqrt(1./sum(A,2)));
x0 = zeros(N,1); x0(pos_idx) = 1; x0(neg_idx) = -1;
beq = zeros(size([0;pos_idx;neg_idx])); beq(2:(1+size(pos_idx)))=1; beq((2+size(pos_idx)):(1+size([pos_idx;neg_idx])))=-1;
Aeq_pos = zeros(size(pos_idx,1),N); Aeq_pos( (pos_idx-1)*size(Aeq_pos,1)+(1:size(Aeq_pos,1))' ) = 1; 
Aeq_neg = zeros(size(neg_idx,1),N); Aeq_neg( (neg_idx-1)*size(Aeq_neg,1)+(1:size(Aeq_neg,1))' ) = 1;
Aeq = [one';Aeq_pos;Aeq_neg];
options = optimoptions('fmincon','Algorithm','trust-region-reflective',...
    'GradObj','on','Hessian','on');
fun = @(x) LaplacianMin(x,L);
f2 = fmincon(fun,x0,[],[],Aeq,beq,[],[],[],options);
subplot(2,2,4);
%scatter(x(f2>0),y(f2>0)); hold on; scatter(x(f2<0),y(f2<0));  scatter(x(f2==0),y(f2==0),[],[0.5,0.5,0.5]);
scatter(x,y,[],colorf(f2,0)); hold on;
scatter(x(pos_idx),y(pos_idx),200,'+','MarkerEdgeColor',[0,0,0],'LineWidth',3); scatter(x(neg_idx),y(neg_idx),200,'x','MarkerEdgeColor',[0,0,0],'LineWidth',3); hold off;
title('spectral w/ labels');

