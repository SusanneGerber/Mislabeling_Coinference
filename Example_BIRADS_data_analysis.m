%
% Robust co-inference of mislabeling risks reveals new patterns
% in breast cancer diagnostics rules and in genomic healthy ageing factors
%
% Susanne Gerber, Lukas Pospisil, Charlotte Hewel, Manuel Rueda, Ali Torkamani, and Illia Horenko
% Published under MIT License, 2017
%

clear all 
close all

%% SET PARAMETERS OF COMPUTATION
% set name of file with data
filename_data = 'data/mammographic_masses.txt';

% set type of used GLM function (see folder 'glm_function')
GLM_function = 'glm_logit2';
%GLM_function = 'glm_linear2'; 

% define risk levels
risk{1} = 0;
risk{2} =  [0 0.001 0.1 0.15];
%risk{2} = [0:0.005:0.04 0.041:0.005:0.1 0.105:0.0025:0.15 0.16:0.02:0.28];
%risk{2} = [0 0.001 0.001*2^2 0.001*2^3 0.001*2^4 0.001*2^5 0.001*2^6 0.001*2^7];

% which cohort to visualise
risk_cohort_visualise = 2;

% set regularisation constants (different values of l1 constraints)
C = [1 2 10 15];
%C = [1:0.5:7 10 15 1e5];
%C = [5e4 1e4 5000 1000 100 50 30 20 18 15 10 5];

% set the proportion of the data used for training
% the rest of the data is used for validation
train_proportion = 3/4;
%train_proportion = 9/10;
%train_proportion = 2/3;

% set number of bootstrap steps
N_bootstrap = 2;
%N_bootstrap = 500;

% number of annealing steps, max number of iterations, and stopping tolerance 
% in optimization problem solver
alg_anneal = 5;
alg_maxit = 500;
alg_tol = 1e-10;

% set number of parallel workers
runparallel = false; % run in parallel or not
numWorkers = 8; % set number of workers
%numWorkers = 12;

% set number of computations for one sequential block
% - each block will run ParallelBlockSize independent threads
% - with respect to this, the input data has to be copied into
%   ParallelBlockSize independent copies - to be sure that there will be
%   not waiting time for accessing the memory
ParallelBlockSize = 100;

%% SET ENVIRONMENT
addpath('data'); % functions for data manipulation
addpath('glm_function'); % GLM functions (linear, logit)
addpath('common'); % common functions
addpath('solver'); % solvers implementation
addpath('postprocess'); % functions for plotting results
addpath('timer'); % functions for measuring time

% restart random generator
%randn('seed',1); % old Matlab style
%rand('seed',1);
rng('default');
crng = rng;
cdate = date();

% initialize parallel computation
if runparallel
    % check if parallel pool exists
    if ~isempty(gcp)
        poolobj = gcp;
        numWorkers = poolobj.NumWorkers;
    else
        poolobj = parpool(numWorkers);
    end
end

% prepare timers for measuring time
global timers
timers = Timers();

% start to measure time for whole run of application
timers.start('all')

%% LOAD MAMMOGRAPHIC DATA
% XX - data (see label for description)
% YY - benign = 0/malignant = 1
% ZZ - 
% label - decription of XX
timers.start('load')
[XX, YY, ZZ, label] = load_mmdata(filename_data);

% split data with respect to benign = {1}/malignant = {2}
% size(Y{1},1) - number of benign, size(Y{2},1) - number of malignant
[X,Y,Z] = split_mmdata(XX, YY, ZZ);
timers.stop('load')

%% INITIALIZE COMPUTATION
% get number of cohorts (in this case {benign/malignant} = 2)
N_cohorts = numel(Y);

% get number of records (number of benign)
N_X = size(X{1},2);

% prepare parallel computation
% create arrays of all possible combinations of C and risk
% and appropriate matrix with original indexes for future reference
%TODO: this indexes can be computed, highly dependent on way how "combvec" works
% the number of columns of this matrix represents the amount of parallel work
% each column defines the combination of parameters [C;risk{1};risk{2};...]
Crisk_comb = combvec(C,risk{1},risk{2}); %TODO: write "combcell" for general numel(risk)?
Crisk_comb_idx = combvec(1:length(C),1:length(risk{1}),1:length(risk{2}));

% maybe the ParallelBlockSize is lower then actual amount of work
% if it is a case, that we modify this number in a such way that we will
% run only one Block which will include all work
ParallelBlockSize = min(ParallelBlockSize, size(Crisk_comb,2));

% compute number of sequential blocks
N_Block = ceil(size(Crisk_comb,2)/ParallelBlockSize); 

% prepare input data for parallel computation
% - here we can fill data for each thread which are independent of the particular
%   value of C, risk, and bootstrap permutation
in = cell(ParallelBlockSize,1);
for idx_thread = 1:ParallelBlockSize
    in{idx_thread}.GLM_function = GLM_function;
    in{idx_thread}.alg_anneal = alg_anneal; % number of annealing steps
    in{idx_thread}.alg_maxit = alg_maxit; % max number of iterations
    in{idx_thread}.alg_tol = alg_tol; % stopping tolerance
    in{idx_thread}.alg_print = false; % print information of optimization algorithm
end

% prepare output data for parallel computation
out = cell(ParallelBlockSize,1);

%% MAIN COMPUTATION
% for every: regularisation parameter, combination of cohorts, risk level, bootstrap
% solve the problem and store results
timers.start('boot')

% initialize variables for results
P = zeros(length(C),N_X,length(risk{risk_cohort_visualise}),N_bootstrap);
r = zeros(length(C),N_cohorts,length(risk{risk_cohort_visualise}),N_bootstrap);
BICs = zeros(length(C),length(risk{risk_cohort_visualise}),N_bootstrap);
AICcs = zeros(length(C),length(risk{risk_cohort_visualise}),N_bootstrap);
LogLs = zeros(length(C),length(risk{risk_cohort_visualise}),N_bootstrap);
LogLs_valid = zeros(length(C),length(risk{risk_cohort_visualise}),N_bootstrap);


ind = cell(1,N_cohorts); % indexes of permutations in bootstrap steps
ind_back = cell(1,N_cohorts); % indexes of inverse permutations in bootstrap steps

% training data
N_train = cell(1,N_cohorts);
X_train = cell(1,N_cohorts);
Y_train = cell(1,N_cohorts);
Z_train = cell(1,N_cohorts);

% validation data
X_valid = cell(1,N_cohorts);
Y_valid = cell(1,N_cohorts);
Z_valid = cell(1,N_cohorts);

% run main cycle
for idx_bootstrap = 1:N_bootstrap
    disp(['- bootstrap = ' num2str(idx_bootstrap) ' of ' num2str(N_bootstrap)]);
    
    % perform random permutation of data
    timers.start('permutation')
    for idx_cohort = 1:N_cohorts
        % number of given data in this cohort
        TT = size(Y{idx_cohort},1);

        % compute random permutation indexes
        ind{idx_cohort} = randperm(TT);
        [~,ind_back{idx_cohort}] = sort(ind{idx_cohort});

        % permute original data
        X{idx_cohort} = X{idx_cohort}(ind{idx_cohort},:);
        Y{idx_cohort} = Y{idx_cohort}(ind{idx_cohort},:);
        Z{idx_cohort} = Z{idx_cohort}(ind{idx_cohort},:);
        
        % prepare training data
        N_train{idx_cohort} = round(train_proportion*TT);
        X_train{idx_cohort} = X{idx_cohort}(1:N_train{idx_cohort},:);
        Y_train{idx_cohort} = Y{idx_cohort}(1:N_train{idx_cohort},:);
        Z_train{idx_cohort} = Z{idx_cohort}(1:N_train{idx_cohort},:);
        
        % prepare validation data
        X_valid{idx_cohort} = X{idx_cohort}((1+N_train{idx_cohort}):end,:);
        Y_valid{idx_cohort} = Y{idx_cohort}((1+N_train{idx_cohort}):end,:);
        Z_valid{idx_cohort} = Z{idx_cohort}((1+N_train{idx_cohort}):end,:);

    end
    timers.stop('permutation')
    
    % solve the problem for largest risk level
    % for every regularisation parameter
    timers.start('boot_Cinit')
    
    % initialize variables
    out_largest_risk = cell(length(C),1); % here we store outputs of DiscreteGLM_cohort
    LogL = zeros(length(C),1); % value of log-likelihood function for every C
    LogL_valid = zeros(length(C),1); % value of log-likelihood function for every C computed on validation data
    AICc = zeros(length(C),1); % values of Akaike information criterion for every C
    BIC = zeros(length(C),1); % value of Bayesian information criterion for every C

    disp(' - precomputing with largest risk value');
    for idx_C = 1:length(C) % though all values of C
        disp(['  - C_idx = ' num2str(idx_C) '; regularization constant C(C_idx) = ' num2str(C(idx_C))]);

        % prepare input variables for DiscreteGLM_cohort algorithm
        in_largest_risk = in{1}; % initialize structure for new input values for DiscreteGLM_cohort
        if idx_C > 1
            % reuse solution from "previous" constrain for initial
            % approximation
            in_largest_risk.P_init = out_largest_risk{idx_C-1}.P;
        else
            % the problem has not been computed yet, there is nothing to
            % reuse as initial approximation
            in_largest_risk.P_init = [];
        end
        in_largest_risk.Y = Y_train;  
        in_largest_risk.X = X_train;
        in_largest_risk.eps1 = C(idx_C); % this particular constraint value
        for idx_cohort = 1:N_cohorts
            in_largest_risk.risk{idx_cohort} = max(risk{idx_cohort});
        end
        in_largest_risk.Y_valid = Y_valid;
        in_largest_risk.X_valid = X_valid;
        
        % call the main solver
        out_largest_risk{idx_C} = DiscreteGLM_cohort(in_largest_risk);
        
        % process output values
        LogL(idx_C) = out_largest_risk{idx_C}.LogL; % value of log-likelihood function
        LogL_valid(idx_C) = out_largest_risk{idx_C}.LogL_valid;
        AICc(idx_C) = out_largest_risk{idx_C}.AICc; % value of Akaike information criterion
        BIC(idx_C) = out_largest_risk{idx_C}.BIC; % value of Bayesian information criterion
    end
    timers.stop('boot_Cinit');
    
    % solve problem with various risk level and C
    timers.start('boot_Cblock')

    % copy data of this bootstrap step into independent copies for every thread
    % in this preprocessing we copy data independent of choise C and risk
    for idx_thread = 1:ParallelBlockSize
        in{idx_thread}.Y = Y_train;
        in{idx_thread}.X = X_train;
        in{idx_thread}.Y_valid = Y_valid;
        in{idx_thread}.X_valid = X_valid;
    end
    
    % run sequentially through the blocks of parallel computation
    disp(' - main computation');
    for idx_Block = 1:N_Block
        disp(['  - Block = ' num2str(idx_Block) ' of ' num2str(N_Block)])

        % how many threads we will run for this block?
        % check if this number is not larger than the whole amount of work
        N_thread = min(ParallelBlockSize,size(Crisk_comb,2) - (idx_Block-1)*ParallelBlockSize);
        
        % - before we run parallel computation, it remains to copy initial
        %   approximations - we will use the solutions from largest risks
        %   from previous computation
        % - the choise of this solution depends (of course) on C with which 
        %   we will compute particular thread
        % - since C (or equivalently idx_C) depends on both of idx_Block 
        %   and idx_thread, it has to be set after setting idx_Block and 
        %   can run before parfor
        % - we will also compute and store idx_C and idx_risk
        for idx_thread = 1:N_thread
            % index of thread with respect to all threads from all blocks
            idx_thread_global = (idx_Block-1)*ParallelBlockSize + idx_thread;

            % get and store original indexes of C and risk from Crisk_comb global index
            in{idx_thread}.i = Crisk_comb_idx(risk_cohort_visualise+1,idx_thread_global);
            in{idx_thread}.j = Crisk_comb_idx(1,idx_thread_global);
            
            % store values from Crisk_comb (or we can use C(idx_C) and risk{...}(risk_idx))
            in{idx_thread}.eps1 = Crisk_comb(1,idx_thread_global);
            for idx_cohort = 1:N_cohorts
                in{idx_thread}.risk{idx_cohort} = Crisk_comb(idx_cohort+1,idx_thread_global);
            end
        
            % reuse solution from previous computation with largest risk
            % value as initial approximation
            in{idx_thread}.P_init = out_largest_risk{idx_C}.P;
        end
        
        % data for parallel threads are prepared, run parallel computation
        if runparallel
            parfor idx_thread = 1:N_thread
                % index of thread with respect to all threads from all blocks
                idx_thread_global = (idx_Block-1)*ParallelBlockSize + idx_thread;        
                
                % run optimization algorithm
                out{idx_thread} = DiscreteGLM_cohort(in{idx_thread});

                % copy coordinates of this problem to output (for postprocessing)
                out{idx_thread}.i = in{idx_thread}.i;
                out{idx_thread}.j = in{idx_thread}.j;
                out{idx_thread}.index = idx_thread_global;
            end
        else
            for idx_thread = 1:N_thread
                disp(['   - Thread = ' num2str(idx_thread) ' of ' num2str(N_thread)])
                
                idx_thread_global = (idx_Block-1)*ParallelBlockSize + idx_thread;        
                out{idx_thread} = DiscreteGLM_cohort(in{idx_thread});
                out{idx_thread}.i = in{idx_thread}.i;
                out{idx_thread}.j = in{idx_thread}.j;
                out{idx_thread}.index = idx_thread_global;
            end
        end
        
        % process results from this block
        % in next block, we will reuse storage of out{..}, therefore it is
        % necessary to move results somewhere else
        for idx_thread = 1:N_thread
            idx_thread_global = (idx_Block-1)*ParallelBlockSize + idx_thread;        
            
            P(out{idx_thread}.j,:,out{idx_thread}.i,idx_bootstrap) = out{idx_thread}.P;
            r(out{idx_thread}.j,:,out{idx_thread}.i,idx_bootstrap) = out{idx_thread}.r;

            BICs(out{idx_thread}.j,out{idx_thread}.i,idx_bootstrap) = out{idx_thread}.BIC;
            AICcs(out{idx_thread}.j,out{idx_thread}.i,idx_bootstrap) = out{idx_thread}.AICc;
            LogLs(out{idx_thread}.j,out{idx_thread}.i,idx_bootstrap) = out{idx_thread}.LogL;
            LogLs_valid(out{idx_thread}.j,out{idx_thread}.i,idx_bootstrap) = out{idx_thread}.LogL_valid;
        end
        
    end
    timers.stop('boot_Cblock');
    
    % save results
    save result/birads_intermediate_9_10
end
% main computation part is done
timers.stop('boot')

%% POSTPROCESSING THE RESULTS

% compute mean values of results through bootstraps
[mean_AICc, mean_LogL, mean_LogL_valid] = compute_mean( AICcs, LogLs, LogLs_valid  );

% - plot the mean value of LogL_valid as a function of regularization
%   parameter C and risk level
% - highlight the position of minimum of this function 
plot_mean(C,risk{risk_cohort_visualise}, mean_LogL_valid, 'mean value of LogL_{valid}', [0.3:0.003:0.7 0.8 1 1.2]);

% - plot the mean value of AICc as a function of regularization
%   parameter C and risk level
% - highlight the position of minimum of this function 
plot_mean(C,risk{risk_cohort_visualise}, mean_AICc, 'mean value of AICc', [450:3.5:630 650 800 1000 1200]);

% find the best P for every bootstrap step with respect to LogLs_valid
P_opt = find_Popt( P, LogLs_valid);

% find the P for every bootstrap step which corresponds to largest regularisation parameter
[~,Cidx_max] = max(C); % get index of largest regularisation parameter
P_opt0 = find_P_Cidx(P, Cidx_max);

% plot histogram of the last component of optimal parameters
figure;
hold on
title('histogram of last component of optimal parameter (with respect to LogLs_valid) through bootstraps')
histogram(P_opt(end,:),8)
hold off

% plot values of optimal parameters for each bootstrap
plot_P(P_opt, label, 'value of optimal parameters for each bootstrap step');

% plot values of parameters corresponding to largest C for each bootstrap
plot_P(P_opt0, label, ['value of parameters corresponding to C=' num2str(max(C)) ' for each bootstrap step']);

% compute mean value of optimal parameters
mean_P = mean(P_opt(1:N_X,:),2)';
mean_P0 = mean(P_opt0(1:N_X,:),2)';

% compute AUC
[ A, Aci, Az, Aciz ] = compute_auc( mean_P, X, Y, Z, GLM_function );

% compute a posterior error
compute_posterior_AICc( mean_AICc, C, risk{risk_cohort_visualise} );
compute_posterior_P( mean_P, P_opt );

% compute impact
[ Impact, Impact_mean, Impact_conf_int ] = compute_impact( mean_P, X, GLM_function );

% print informations about impact
print_impact( Impact_mean, Impact_conf_int, label );

% compute histogram for mean risk age
[AgeIntervals,mean_risk_age,x_mi,x_pl] = compute_mean_risk_age(AICcs, P_opt, GLM_function);

% plot "Mean risk age"
plot_mean_risk_age( AgeIntervals,mean_risk_age,x_mi,x_pl );


%% FINISH COMPUTATION
timers.stop('all')
if runparallel
%    delete(poolobj);
end

%% PRINT TIMERS
fprintf('\n')
timers.print()
