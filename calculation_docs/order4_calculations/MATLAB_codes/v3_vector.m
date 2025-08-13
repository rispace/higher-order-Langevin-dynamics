% This MATLAB script computes the fourth component vector of the
% 4th order Langevin dynamics

% The symbols requred
% Here, A is a dxd matrix, d is the dimension of the data;
% b, theta, v1, v2, v3 are all d dimensional vectors.

syms A b theta v1 v2 v3 eta gamma k r s t w y z mu30 mu31 mu32 mu33

% Define the potential function
nabla_f = @(theta) A*theta -b;

% Compute T1
T1 = int(int(nabla_f(theta+(z-k*eta)*v1),z,k*eta,y),y,k*eta,w);

% Compute T2
T2 = gamma*v2*(w-k*eta)^2/factorial(2)+gamma^2*(-v1+v3)*(w-k*eta)^3/factorial(3);

% Expression to compute T3
exp1 = exp(-gamma*((k+1)*eta-s))*nabla_f(theta+(z-k*eta)*v1);
% Compute T3
T3 = gamma^4*int(int(int(int(int(exp1,z,k*eta,y),y,k*eta,w),w,k*eta,r),r,k*eta,s),s,k*eta,(k+1)*eta);

% Expression to compute T4
exp2 = exp(-gamma*((k+1)*eta-s))*exp(-gamma*(r-w))*nabla_f(theta+(z-k*eta)*v1);
% Compute T4
T4 = gamma^4*int(int(int(int(int(exp2,z,k*eta,y),y,k*eta,w),w,k*eta,r),r,k*eta,s),s,k*eta,(k+1)*eta);

% Compute T5
T5 = mu30*theta+mu31*v1+mu32*v2+mu33*v3;

% Expression to compute m3
exp3 = exp(-gamma*((k+1)*eta-s))*nabla_f(theta+(w-k*eta)*v1-T1+T2);
% Compute m3
m3 = -gamma^2*int(int(int(exp3,w,k*eta,r),r,k*eta,s),s,k*eta,(k+1)*eta) + T3+T4+T5;


latex(m3)
