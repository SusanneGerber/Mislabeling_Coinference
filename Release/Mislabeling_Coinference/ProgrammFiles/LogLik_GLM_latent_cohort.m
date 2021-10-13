function [LogL,dLogL] = LogLik_GLM_latent_cohort(x,X,Y,N_X,TT,GLM_function,N_cohorts,TT_full)


% the epsilon used for avoiding infinity

myeps = 1e-13;
P=x(1:N_X);%tau=x(2*N_X);
risk=x(2*N_X:2*N_X+N_cohorts-1);
LogL=0;
dLogL=zeros(1,2*N_X-1+N_cohorts);
R=[1-risk(1) risk(2);risk(1) 1-risk(2)];


for n=1:N_cohorts

%% Here The function is evaluated -> fit to data -> made prediction 
        [phi1_s,dphi1_s]=feval(GLM_function,P,X{n});

        
   
    temp = max((R(1,1)*(1-phi1_s).*(1-Y{n}) + (1-R(1,1))*phi1_s.*(1-Y{n})...
        +(1-R(2,2))*(1-phi1_s).*Y{n}+R(2,2)*phi1_s.*Y{n}),myeps);

    LogL=LogL + sum(log(temp));

    dLogL(1:N_X) = dLogL(1:N_X) + ...
            sum(...
                bsxfun(@times,...
                -R(1,1)*bsxfun(@times,dphi1_s,1-Y{n}')...
                      +(1-R(1,1))*bsxfun(@times,dphi1_s,1-Y{n}') ...
                      - (1-R(2,2))*bsxfun(@times,dphi1_s,Y{n}')...
                      +R(2,2)*bsxfun(@times,dphi1_s,Y{n}')...
                   ,1./temp')...
            ,2)';



end

LogL=-(1/(TT_full))*LogL;
dLogL=-(1/(TT_full))*dLogL;
end

