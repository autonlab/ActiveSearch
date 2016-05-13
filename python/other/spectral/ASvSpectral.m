% params
K = [1,0];
N=100; 
s=0.1; 

% two moons
x1 = 2*rand(N,1)-1; x2 = 2*rand(N,1); 
y1=normrnd(sqrt(1-x1.*x1),s); y2=normrnd(-sqrt(2*x2-x2.*x2)+0.5,s);
x= [x1;x2]; y=[y1;y2];
T= [ones(N,1);zeros(N,1)];

close all;
Compare(x,y,T,s,K(1),K(2));

% two spirals
N = 200;
s = 0.05;
theta1 = 0.5*pi+rand(N,1)*(3*pi-0.5*pi); theta2 = 0.5*pi+rand(N,1)*(3*pi-0.5*pi);
x1 = normrnd(cos(theta1).*theta1,s*theta1); y1 = normrnd(sin(theta1).*theta1,s*theta1);
x2 = normrnd(cos(theta2+pi).*theta2,s*theta2); y2= normrnd(sin(theta2+pi).*theta2,s*theta2);
x= [x1;x2]; y=[y1;y2];
T= [ones(N,1);zeros(N,1)];

figure();
Compare(x,y,T,mean(s*theta1),K(1),K(2));


