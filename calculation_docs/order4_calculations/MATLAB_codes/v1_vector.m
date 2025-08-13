% This MATLAB script computes the second component vector of the
% 4th order Langevin dynamics

% The symbols requred
% Here, A is a dxd matrix, d is the dimension of the data;
% b, theta, v1, v2, v3 are all d dimensional vectors.

syms A b theta v1 v2 v3 eta gamma k r s t w y z mu10 mu11 mu12 mu13

% Define the potential function
nabla_f = @(theta) A*theta -b;

g1 = theta + (w-k*eta)*v1;
g2 = theta + (s-k*eta)*v1;

% Compute T1
T1 = int(int(nabla_f(g1),w,k*eta,r),r,k*eta,s);

% Compute T2
T2 = gamma*v2*(s-k*eta)^2/2 + gamma^2*(-v1+v3)*(s-k*eta)^3/6;

% Compute T3
T3 = gamma^2*int(int(int(nabla_f(g1),w,k*eta,r),r,k*eta,s),s,k*eta,(k+1)*eta);

% Compute T4
T4 = mu10*theta+mu11*v1+mu12*v2+mu13*v3;

% Compute the second component v1 (denoting m1 to avoid conflict)
m1 = - int(nabla_f(g2-T1+T2),s,k*eta,(k+1)*eta) + T3 +T4;

m1_simplified = simplify(m1);

latex(m1_simplified)
