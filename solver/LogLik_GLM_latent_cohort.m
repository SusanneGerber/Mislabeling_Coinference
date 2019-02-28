function [LogL,dLogL_P,dLogL_r] = LogLik_GLM_latent_cohort(P,r,X,Y,GLM_function)
%LOGLIK_GLM_LATENT_COHORT compute the value of Log likelihood objective function
%   [LogL,dLogL_P,dLogL_r] = LOGLIK_GLM_LATENT_COHORT(P,r,X,Y,GLM_function)
%   Compute the value of Log likelihood function 
%          LogL(P,r) = sum_cohorts sum_T log(r1*phi1*Y1 + r2*phi2*Y2)
%   with gradients with respect to variables dLogL_P,dLogL_r
%
%  INPUT:
%   P - values of model parameters
%   r - risk values (for each cohort one value, the second is computed as 1-r)
%   X - given data (X{n} corresponds to n-th cohort, each row represents one measurement)
%   Y - labels of the data (same layout as X)
%   GLM_function - type of used generalised linear model (GLM), see folder "glm_function"
%
%  OUPUT:
%   LogL - value of Log likelihood function
%   dLogL_P - column vector of gradient of Log likelihood function with respect to P
%   dLogL_r - column vector of gradient of Log likelihood function with respect to r
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%

% the epsilon used for avoiding infinity
myeps = 1e-13;

N_cohorts = numel(X); % number of cohorts
N_P = length(P); % number of parameters
N_r = N_cohorts; % number of risk values

% allocate (initialize) return variables
LogL = 0; % function value
dLogL_P = zeros(N_P,1); % gradient ( with respect to P)
dLogL_r = zeros(N_r,1); % gradient ( with respect to r)

% go through cohorts and add contribution
for n=1:N_cohorts
    % evaluate GLM_function in given parameters
    % phi1_s - column vector of function values in every row of X{n}
    % dphi1_s - matrix with gradients - gradient in every row of X{n} is
    %           stored in column
    [phi1_s,dphi1_s]=feval(GLM_function,P,X{n});

    % compute auxiliary vector - arguments of logarithms in LogLik - convex
    % combination of phi1(t),phi2(t) with coefficients r1(t),r2(t) multiplied by Y
    % in our case:
    %   phi2 = (1 - phi1)
    %   r1 = r(n)
    %   r2 = (1 - r1)
    %   Y2 = (1 - Y1) (indicator function)
    % maybe (numerics) argument of logarithm (denoted as temp) is 0 or 1, 
    %  in that case put here myeps instead of 0
    temp = min(max(r(n)*phi1_s.*Y{n} + (1-r(n))*(1-phi1_s).*(1-Y{n}),myeps),1-myeps);
    
    % add increment of function value with respect to this cohort
    LogL=LogL + sum(log(temp));

    % add increment of gradient with respect to parameters P
    % in general : d{log(f(P))} = (1/f(P))*df(P)
    % in our case: d{log(temp(P))} = (1/temp)*(r1*dphi1*Y1 + r2*dphi2*Y2)
    % dphi2 = d(1-phi1) = -dphi1
    dLogL_P = dLogL_P + ...
            sum(...
                bsxfun(@times,...
                      r(n)*bsxfun(@times,dphi1_s,Y{n}') ...
                      - (1-r(n))*bsxfun(@times,dphi1_s,1-Y{n}')...
                   ,1./temp')...
            ,2);

    % add increment of gradient with respect to risk values r
    % in general : d{log(f(r))} = (1/f(r))*df(r)
    % in our case: d{log(temp(r))} = (1/temp)*(phi1*Y1 - phi2*Y2)
    % dr2 = d(1-r1) = -dr1 = -1
    dLogL_r(n) = dLogL_r(n) + ...
            sum(...
                (1./temp).*(phi1_s.*Y{n} - (1 - phi1_s).*(1 - Y{n}) )...
            );

end

% compute the number of all measurements
TT_full = 0;
for i=1:numel(Y)
   TT_full = TT_full + length(Y{i}); 
end

% scale (and multiply with -1 since we will use minimization solver)
coeff = -(1/(TT_full));
LogL = coeff*LogL;
dLogL_P= coeff*dLogL_P;
dLogL_r= coeff*dLogL_r;

end

