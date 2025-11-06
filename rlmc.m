%% Strong convergence of randomized LMC vs classic LMC (reference) on a Gaussian mixture
clear; clc; rng(1);

%% ===== Mixture target: set externally =====
% Dimension is inferred from the length of mu1/mu2
mu1 = [-2, -2];                    % 1-by-d
mu2 = [ 2,  2];                    % 1-by-d
s1  = 1.0;                         % scalar std of component 1
s2  = 1.0;                         % scalar std of component 2
p1  = 0.5; p2 = 0.5;               % weights (sum to 1 recommended)
d   = numel(mu1);

%% ===== Global experiment params =====
T  = 1.0;
N  = 2^15;               % finest grid for reference
dt = T / N;
M  = 500;                % number of i.i.d. trajectories for Monte Carlo
R  = [1, 16, 32, 64, 128, 256, 512];   % refinement factors (R(1)=1 used as reference)

%% ===== Pre-generate the driving Brownian path on the finest grid =====
% Common Random Numbers: ALL methods/stepsizes use these same fine increments.
dW_fine = sqrt(dt) * randn(M, N, d);    % size M x N x d

%% ===== Helper: gradient of U for 2-component isotropic mixture (log-sum-exp stable) =====
gradU = @(X) gradU_mog_weighted(X, mu1, mu2, s1, s2, p1, p2);  % X: M-by-d

%% ===== Reference solution: classic LMC (Euler¨CMaruyama) on the finest grid =====
Xref = zeros(M, d);                     % initial x0 = 0
sqrt2 = sqrt(2);
for n = 1:N
    dWn = squeeze(dW_fine(:, n, :));    % M-by-d
    G   = gradU(Xref);                  % M-by-d
    Xref = Xref - G * dt + sqrt2 * dWn; % classic LMC step
end
Xref_T = Xref;                           % reference at t=T

%% ===== Randomized LMC on coarser stepsizes, coupled with the SAME Brownian path =====
P = numel(R);
ErrMS = zeros(1, P-1);                   % mean-square errors for p=2..P

for p = 2:P
    Dt = R(p) * dt;                      % coarse step size
    L  = N / R(p);                       % number of coarse steps
    Y  = zeros(M, d);                    % start from 0, same as reference
    
    for j = 1:L
        % Brownian increment on [t_{j-1}, t_j]: sum of fine increments (CRN coupling)
        idx = (R(p)*(j-1)+1) : (R(p)*j);
        Winc = squeeze(sum(dW_fine(:, idx, :), 2));      % M-by-d, Var = Dt
        
        % Randomization: tau ~ U(0,1), Brownian-bridge noise Z2 ~ N(0, I_d)
        tau  = rand(M, 1);                               % M-by-1
        Z2   = randn(M, d);                              % M-by-d
        Wtau = tau .* Winc + sqrt(Dt * 1.0) .* (sqrt(tau .* (1 - tau)) .* Z2);
        
        % Stage 1: Y^{tau} = Y - gradU(Y) * (tau*Dt) + sqrt(2) * Wtau
        GY   = gradU(Y);                                 % M-by-d
        Ytau = Y - (tau .* Dt) .* GY + sqrt2 * Wtau;
        
        % Stage 2: Y_{j} = Y - gradU(Y^{tau}) * Dt + sqrt(2) * Winc
        GYt  = gradU(Ytau);
        Y    = Y - Dt * GYt + sqrt2 * Winc;
    end
    
    % Strong error at terminal time (mean of squared Euclidean distance)
    diff   = Y - Xref_T;                       % M-by-d
    ErrMS(p-1) = mean(sum(diff.^2, 2));        % E[||Y(T)-Xref(T)||^2]
end

%% ===== Plot & slope fit =====
Dtvals   = dt * R(2:end);
sqrt_err = sqrt(ErrMS);

loglog(Dtvals, sqrt_err, 'md-', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
loglog(Dtvals, 0.11 * sqrt(Dtvals), 'r', 'LineWidth', 1.2); hold on;   % order 1/2 guide
loglog(Dtvals, 0.2  * (Dtvals),     'k--', 'LineWidth', 1.2);          % order 1 guide
grid on; box on;
xlabel('Stepsizes \Delta t');
ylabel('Mean-square errors^{1/2}');
title('Strong error of randomized LMC vs classic LMC reference');
legend('RMSE', 'order 0.5', 'order 1.0', 'Location', 'SouthEast');

A   = [ones(numel(Dtvals),1), log(Dtvals(:))];
rhs = log(sqrt_err(:));
sol = A \ rhs; q = sol(2);                 % fitted slope
resid = norm(A*sol - rhs);
fprintf('Fitted slope q ¡Ö %.3f, residual %.3e\n', q, resid);

%% ====== ---- Local function: gradient of mixture potential ---- ======
function G = gradU_mog_weighted(X, mu1, mu2, s1, s2, p1, p2)
% X: M-by-d, mu*: 1-by-d, s*: scalar
% ?U(x) = a1(x)*(x-mu1)/s1^2 + a2(x)*(x-mu2)/s2^2,  a_k are responsibilities with weights p_k.
    [M, d] = size(X);
    Xc1 = X - mu1;                          % M-by-d
    Xc2 = X - mu2;                          % M-by-d
    % log components up to an additive constant: log p_k - ||x-¦Ì_k||^2/(2 s_k^2) - d log s_k
    l1  = log(p1) - 0.5 * sum(Xc1.^2, 2) / (s1^2) - d * log(s1);   % M-by-1
    l2  = log(p2) - 0.5 * sum(Xc2.^2, 2) / (s2^2) - d * log(s2);   % M-by-1
    m   = max(l1, l2);                     % M-by-1, for log-sum-exp stability
    e1  = exp(l1 - m); e2 = exp(l2 - m);
    a1  = e1 ./ (e1 + e2);                 % responsibilities
    a2  = 1 - a1;
    G   = (a1 ./ (s1^2)) .* Xc1 + (a2 ./ (s2^2)) .* Xc2;  % M-by-d
end
