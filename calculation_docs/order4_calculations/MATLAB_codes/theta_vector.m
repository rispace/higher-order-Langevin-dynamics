% This MATLAB script computes the first component vector of the
% 4th order Langevin dynamics

% The symbols requred
% Here, A is a dxd matrix, d is the dimension of the data;
% b, theta, v1, v2, v3 are all d dimensional vectors.

syms A b theta v1 v2 v3 eta gamma k r s t w y z mu00 mu01 mu02 mu03

% Define the potential function
nabla_f = @(theta) A*theta -b;

g = theta+(y-k*eta)*v1;

% Compute T1
T1 = int(int(nabla_f(g),y,k*eta,w),w, k*eta,r);

% Compute T2
T2 = gamma*v2*(((r-k*eta)^2)/2)+gamma^2*(-v1 + v3)*(((r-k*eta)^3)/6);

% Compute T3
T3 = gamma^2*int(int(int(int(nabla_f(g),y,k*eta,w),w,k*eta,r),r,k*eta,s),s,k*eta,(k+1)*eta);

% Compute T4
T4 = mu00*theta+mu01*v1+mu02*v2+mu03*v3;

% Compute theta (denote as m0 to avoid conflict)
m0 = -int(int(nabla_f(theta+(r-k*eta)*v1-T1+T2),r,k*eta,s),s,k*eta,(k+1)*eta)+T3+T4;

m0_simplified = simplify(m0);


latex(m0_simplified)
