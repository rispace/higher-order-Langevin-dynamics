% This MATLAB script computes the third component vector of the
% 4th order Langevin dynamics

% The symbols requred
% Here, A is a dxd matrix, d is the dimension of the data;
% b, theta, v1, v2, v3 are all d dimensional vectors.

syms A b theta v1 v2 v3 eta gamma k r s t w y z mu20 mu21 mu22 mu23

% Define the potential function
nabla_f = @(theta) A*theta -b;

% Compute T1
T1 = int(int(nabla_f(theta+(y-k*eta)*v1),y,k*eta,w),w,k*eta,r);

% Compute T2
T2 = gamma*v2*(r-k*eta)^2/factorial(2)+gamma^2*(-v1+v3)*(r-k*eta)^3/factorial(3);

% Compute T3
T3 = -gamma^3 * int(int(int(int(nabla_f(theta+(y-k*eta)*v1),y,k*eta,w),w,k*eta,r),r,k*eta,s),s,k*eta,(k+1)*eta);

% Compute T4
exprn = exp(-gamma*(s-r))*nabla_f(theta+(y-k*eta)*v1);
T4 = -gamma^3 * int(int(int(int(exprn,y,k*eta,w),w,k*eta,r),r,k*eta,s),s,k*eta,(k+1)*eta);

% Compute T5
T5 = theta*mu20 +v1*mu21 +v2*mu22 +v3*mu23;

% Compute v2 (aka m2)
m2 = gamma*int(int(nabla_f(theta+(r-k*eta)*v1-T1+T2),r,k*eta,s),s,k*eta,(k+1)*eta)+T3 +T4+T5;

latex(m2)
