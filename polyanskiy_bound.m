%% Example of parameters (for debugging)
k      = 100;
n      = 30000;
Ka     = 200;
EbN0db = 2;
Niq    = 1000;

%% Power and codebook size
P = k*10^(EbN0db/10)/n;
M = 2^k;


%% Initialization
% for the sum over t in Eq. (3)
t_vec = 1:Ka;
% for the optimization over rho and rho1 in E(t)
rho_vec  = linspace(0,1,100); % This goes from 0 to 1 in principle
rho1_vec = linspace(0,1,100); % From 0 to 1
rho      = repmat(rho_vec,length(rho1_vec),1,length(t_vec));
rho1     = permute(repmat(rho1_vec,length(rho_vec),1,length(t_vec)),[2,1,3]);
t        = permute(repmat(t_vec,length(rho1_vec),1,length(rho_vec)),[1,3,2]);

% Some functions for the computation of pt as defined in [1, Th. 1]
R1_f = @(n,M,t)  1./n*log(M)-1./(n*t).*gammaln(t+1);
R2_f = @(n,Ka,t) 1/n*(gammaln(Ka+1)-gammaln(t+1)-gammaln(Ka-t+1));
% D_f  = @(P1,t,rho,rho1) (P1.*t-1).^2 + 4*P1.*t.*(1+rho.*rho1)./(1+rho);
% lambda_f = @(P1,t,rho,rho1) (P1.*t-1+sqrt(D_f(P1,t,rho,rho1)))./(2.*(1+rho1.*rho).*P1.*t);
% mu_f = @(P1,t,rho,rho1) rho.*lambda_f(P1,t,rho,rho1)./(1+P1.*t.*lambda_f(P1,t,rho,rho1));
% %
% a_f = @(P1,t,rho,rho1) rho.*log(1+P1.*t.*lambda_f(P1,t,rho,rho1))+log(1+P1.*t.*mu_f(P1,t,rho,rho1));
% %
% b_f = @(P1,t,rho,rho1) rho.*lambda_f(P1,t,rho,rho1)- mu_f(P1,t,rho,rho1)./(1+P1.*t.*mu_f(P1,t,rho,rho1));
% E0_f = @(P1,t,rho,rho1) rho1.*a_f(P1,t,rho,rho1) + log(1-b_f(P1,t,rho,rho1).*rho1);
% Et_f = @(P1,t,rho,rho1,n,M,Ka) squeeze(max(max(-rho.*rho1.*t.*R1_f(n,M,t) - rho1.*R2_f(n,Ka,t) + E0_f(P1,t,rho,rho1))));
% pt_f   = @(P1,t,rho,rho1,n,M,Ka) exp(-n*Et_f(P1,t,rho,rho1,n,M,Ka));

% for the optimization over P' (here denoted by P1)
P1_vec = linspace(1e-9,P,20);
epsilon = zeros(size(P1_vec));

%% Evaluation of the bound, optimized over P'
% Note: one can perform a golden search for the optimal P'. We do it in the
% evaluation of the RCU bound for Ka random.
for pp = 1:length(P1_vec)
    P1 = P1_vec(pp);
    % compute of p0
    p0 = nchoosek(Ka, 2) / M + Ka * gammainc(n * P / P1, n, 'upper');
    % compute R1, R2
    R1 = 1 / n * log(M) - 1 ./ (n * t) .* gammaln(t+1);
    R2 = 1 / n * (gammaln(Ka + 1) - gammaln(t + 1) - gammaln(Ka - t + 1));
    % compute D
    D  = (P1.*t-1).^2 + 4*P1.*t.*(1+rho.*rho1)./(1+rho);
    % compute lambda_
    lambda_ = (P1.*t-1+sqrt(D))./(2.*(1+rho1.*rho).*P1.*t);
    % compute mu
    mu = rho .* lambda_./(1+P1.*t.*lambda_);
    % compute a, b
    a = rho.*log(1+P1.*t.*lambda_)+log(1+P1.*t.*mu);
    b = rho.*lambda_- mu./(1+P1.*t.*mu);
    % compute E0
    E0 = rho1.*a + log(1-b.*rho1);
    % compute Et
    Et = squeeze(max(max(-rho.*rho1.*t.*R1 - rho1.*R2 + E0)));
    % compute of pt
    pt = exp(-n*Et);

    % compute of qt (for t = 1 only, as in [1])
    It = zeros(1,1000);
    for II = 1:Niq
        Zi = sqrt(.5)*(randn(1,n) + 1i*randn(1,n));
        codebook = sqrt(.5*P1)*(randn(Ka,n) + 1i*randn(Ka,n));
        it   = n*log(1+P1) + (sum(abs(repmat(Zi,Ka,1)+codebook).^2,2)./(1+P1)-sum(abs(repmat(Zi,Ka,1)).^2,2));
        It(II) = min(it);
    end
    [prob,gam] = ecdf(It);
    qt = min(prob + exp(n*(R1_f(n,M,1)+R2_f(n,Ka,1))-gam));
    % Computation of RHS of [1, Eq. (3)] for a given P' < P
    epsilon(pp) = t_vec(1)/Ka*min(pt(1),qt) + t_vec(2:end)/Ka*pt(2:end) + p0;
end

% Take the minimum over P'
epsilon
[epsilon, idx_P1opt] = min([epsilon]);
P1_opt = P1_vec(idx_P1opt);
%
epsilon
P1_opt
