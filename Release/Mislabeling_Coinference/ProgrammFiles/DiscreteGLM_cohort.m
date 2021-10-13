function [out] = DiscreteGLM_cohort(in)
%% Just setting parameters which are going in
%N_times_called_1 = 0;

Y=in.Y;
X=in.X;
Y_valid=in.Y_valid;
X_valid=in.X_valid;
TT=in.TT;
GLM_function=in.GLM_function;
eps1=in.eps1;
risk=in.risk;
xxx_init=in.xxx_init;
N_anneal=in.anneal;
TT_valid=in.TT_valid;

N_cohorts=numel(TT);
TT_full=0;TT_full_valid=0;
for n=1:N_cohorts
    TT_full=TT_full+TT{n};
    TT_full_valid=TT_full_valid+TT_valid{n};
    min_risk{n}=min(risk{n});
    max_risk{n}=max(risk{n});
end
N_X=size(X{1},2);
tol=1e-10;
MaxIter=1500;
N=N_X;
options=optimset('UseParallel','always',...
    'GradObj','on','Algorithm','sqp','MaxIter',MaxIter,'Display','off','TolFun',tol,'TolCon',tol,'TolX',tol);
n_x=N_X*2-1+N_cohorts;

%% Inequality constraints matrices -> just filling em up i guess?
A1=zeros(N_X-1,n_x);
A2=zeros(N_X-1,n_x);
A3=zeros(N_X-1,n_x);
for i=1:(N_X-1)
    A1(i,i+1)=1; A1(i,N_X+i)=-1;
    A2(i,i+1)=-1; A2(i,N_X+i)=-1;
    A3(i,N_X+i)=-1;
end

b=[zeros(3*(N_X-1),1);eps1];
A=[A1;A2;A3;zeros(1,N_X) ones(1,N_X-1) zeros(1,N_cohorts)];
Aeq=[];beq=[];
for n=1:N_cohorts
    if min_risk{n}==max_risk{n}
        Aeq=[Aeq;zeros(1,2*N_X-1) n==1:N_cohorts];
        beq=[beq;min_risk{n}];
    else

    end 
end

%% Number of annealing steps (against over/under fitting i guess)
for n_anneal=1:N_anneal
    if ~and(n_anneal==1,~isempty(xxx_init))
        clear xxx_init
        xxx_init(1:N_X)=randn(1,N_X);xxx_init(2:N_X)=xxx_init(2:N_X)./sum(abs(xxx_init(2:N_X)))*max_risk{n};
        xxx_init(N_X+1:2*N_X-1)=abs(xxx_init(2:N_X));
        for n=1:N_cohorts
            xxx_init=[xxx_init max_risk{n}];
        end
    end


    [xxx0,fff,flag,output] =  fmincon(@(x)LogLik_GLM_latent_cohort...
        (x,X,Y,N_X,TT,GLM_function,N_cohorts,TT_full)...
        ,xxx_init,(A),(b),Aeq,beq,[],[],[],options);
    

    if n_anneal==1
        Lfin=fff;
        Pfin=xxx0;
    else
        if Lfin>fff
            Lfin=fff;
            Pfin=xxx0;
        end
    end
end
%% Just output
if TT_full_valid>0
out.LogL_valid=LogLik_GLM_latent_cohort(Pfin,X_valid,Y_valid,N_X,TT_valid,GLM_function,N_cohorts,TT_full_valid);
else
out.LogL_valid=-1e12;
end

out.P=Pfin;
out.N_par=length(find(abs(Pfin([1:N_X+1 2*N_X:(2*N_X+1)]))>tol));%]))>tol));% 
out.LogL=-Lfin*(TT_full);
out.BIC=-2*out.LogL+2*out.N_par*log(TT_full);
out.AICc=-2*out.LogL+2*out.N_par+2*out.N_par*(out.N_par+1)/(TT_full-out.N_par-1);
