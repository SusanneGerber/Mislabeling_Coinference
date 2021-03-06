function [out] = DiscreteGLM_cohort(in)
%DISCRETEGLM_COHORT solve optimization problem for given set of parameters
%   [out] = DISCRETEGLM_COHORT(in)
%
%  INPUT:
%   in - structure including input parameters and data
%     in.X - matrix of data, each row corresponds to one record
%     in.Y - labels of data (cohort attribution)
%     in.X_valid - validation data (used only for computation of out.LogL_valid)
%     in.Y_valid - labels of validation data
%     in.GLM_function - type of GLM function
%     in.eps1 - regularisation constants (l1 constraint ||P||_1 <= C)
%     in.risk - risk level (= solution r)
%     in.P_init - initial approximation of unknown P
%     in.P_zerotol - tolerance for computing out.N_par
%     in.alg_anneal - number of used annealing steps
%     in.alg_maxit - maximum number of iterations of used optimization algorithm
%     in.alg_tol - stopping criteria for used optimization algorithm
%     in.alg_print - print informations about algorithm performance (true/false)
%
%  OUPUT:
%   out - structure with output results and solution
%     out.P - parameter solution
%     out.r - risk solution
%     out.N_par - number of non-zero values (up to precision) in P
%     out.LogL - value of Log likelihood function (objective function)
%     out.LogL_valid - value of Log likelihood function on validation data
%     out.BIC - value of Bayesian information criterion
%     out.AIC - value of Akaike information criterion
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%

%% READ INPUT VALUES
% maybe some variables are not set, in that case use default settings
if ~isfield(in,'P_zerotol')
    in.P_zerotol = 1e-3;
end
if ~isfield(in,'alg_anneal')
    in.alg_anneal = 5;
end
if ~isfield(in,'alg_maxit')
    in.alg_maxit = 1000;
end
if ~isfield(in,'alg_tol')
    in.alg_tol = 1e-10;
end
if ~isfield(in,'alg_print')
    in.alg_print = true;
end
if ~isfield(in,'P_init')
    in.P_init = [];
end
if ~isfield(in,'risk')
    in.risk = 1e-2;
end

% set variables from input
Y=in.Y;
X=in.X;
Y_valid=in.Y_valid;
X_valid=in.X_valid;
GLM_function=in.GLM_function;
eps1=in.eps1;
P_init=in.P_init;

% risk to r (= fixed,given)
r=zeros(numel(in.risk),1);
for i=1:numel(in.risk)
    r(i) = in.risk{i}; % TODO: in Matlab, there is functions for cell->vec
end

%% SET ADDITIONAL VARIABLES
N_cohorts=numel(Y);

% compute number of all records
% and find maximum and minimum risk level for each cohort
TT_full=0;
TT_full_valid=0;
for n=1:N_cohorts
    TT_full=TT_full+size(Y{n},1);
    TT_full_valid=TT_full_valid+size(Y_valid{n},1);
end

%% PREPARE OPTIMIZATION ALGORITHM
N_X = size(X{1},2);
options=optimset(... %'UseParallel','always',...
    'GradObj','on','Algorithm','sqp','MaxIter',in.alg_maxit,'Display','off','TolFun',in.alg_tol,'TolCon',in.alg_tol,'TolX',in.alg_tol...
    );

% prepare objects of constraints
[A,b,Aeq,beq,lb,ub] = fmincon_constraint(N_X, eps1);

%% RUN MAIN CYCLE - ANNEALING
for n_anneal=1:in.alg_anneal
    % set initial approximation of P
    if or(n_anneal>1,isempty(P_init))
        % if initial approximation is not provided or we are computing new
        % annealing step, then we generate random initial values
        P_init = randn(1,N_X);
    end
    
    % from initial approximations compose initial approximation for fmincon
    x0 = fmincon_x0(P_init);
    
    % run the optimization algorithm
    [x,fff,~,output] =  fmincon(...
        @(x)fmincon_fx(x,r,X,Y,N_X,GLM_function),...
        x0,...
        full(A),(b),Aeq,beq,lb,ub,[],options);
    
    % get P from fmincon solution
    P = fmincon_strip(x, N_X);
    
    % print final info about optimization procedure
    if in.alg_print
        disp(['- it=' num2str(output.iterations) ', err=' num2str(output.firstorderopt) ', fff=' num2str(fff)])
    end
    
    if n_anneal==1
        % this is first annealing step, it has to be the best (for now)
        Lfin=fff;
        Pfin=P;
    else
        % if this annealing step is better than previous then replace old values
        if Lfin>fff
            Lfin=fff;
            Pfin=P;
        end
    end
end

%% SET OUTPUT VARIABLES
% set output solution from the best annealing step
out.P=Pfin;
out.r=r;
out.LogL=-Lfin*(TT_full);

% compute number of non-zero parameters P
out.N_par=length(find(abs(Pfin>in.P_zerotol)));

% compute information criteria
out.BIC=-2*out.LogL+2*out.N_par*log(TT_full);
out.AICc=-2*out.LogL+2*out.N_par+2*out.N_par*(out.N_par+1)/(TT_full-out.N_par-1);

% compute value of Log likelihood function on validation data
if ~isfield(in,'X_valid') || ~isfield(in,'Y_valid')
    % without validation - not enought provided variables
    out.LogL_valid=NaN;
else
    out.LogL_valid=LogLik_GLM_latent_cohort(out.P,out.r,X_valid,Y_valid,GLM_function);
end

end


function [fx,dfx] = fmincon_fx(x,r,X,Y,N_X,GLM_function)
% define objective function for fmincon algorithm

% initialize gradient
dfx = zeros(size(x));

P = fmincon_strip(x, N_X);
[fx,dfx(1:N_X),~] = LogLik_GLM_latent_cohort(...
    P,r,X,Y,GLM_function);

end

function [x0] = fmincon_x0(P0)
% compose initial approximation for fmincon algorithm
if ~isempty(P0)
    x0=[P0,abs(P0(2:end))];
else
    x0 = [];
end
end

function [P] = fmincon_strip(x, N_X)
% extract P from solution of fmincon
P = x(1:N_X);
end

function [A,b,Aeq,beq,lb,ub] = fmincon_constraint(N_X, eps1)
% prepare constraints for fmincon algorithm

n_x=N_X*2-1;

%%% Inequality constraints matrices
e=ones(N_X-1,1);
A1=[spdiags(e,1,N_X-1,N_X), -spdiags(e,0,N_X-1,n_x-N_X)];
A2=[-spdiags(e,1,N_X-1,N_X), -spdiags(e,0,N_X-1,n_x-N_X)];
A=[A1;A2;zeros(1,N_X) ones(1,N_X-1)];

b=[zeros(2*(N_X-1),1);eps1];

%%% Equality constraints matrices
Aeq=[];beq=[];

%%% bound constraints
lb = [-Inf*ones(N_X,1);ones(N_X-1,1)]; % xi is bounded from below
ub = [];

end

