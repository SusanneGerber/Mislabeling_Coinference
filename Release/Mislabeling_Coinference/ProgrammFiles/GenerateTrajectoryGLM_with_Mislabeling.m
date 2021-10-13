function [Y,LogL] = GenerateTrajectoryGLM_with_Mislabeling(x,X,N_X,GLM_function,N_cohorts,TT_full)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% the epsilon used for avoiding infinity
myeps = 1e-13;
P=x(1:N_X);%tau=x(2*N_X);
risk=x(2*N_X:2*N_X+N_cohorts-1);
LogL=0;
R=[1-risk(1) risk(2);risk(1) 1-risk(2)];
for n=1:N_cohorts
%      if n==1
%          R=[1 0;0 1]; 
%      else
%          R=[1-risk(1) risk(2);risk(1) 1-risk(2)];
%      end
        [phi1_s,dphi1_s]=feval(GLM_function,P,X{n});
        T=length(phi1_s);Y{n}=zeros(T,1);
        for t=1:T
            pi_Y=cumsum(R*[1-phi1_s(t);phi1_s(t)])';
            [~,mm_min]=find(pi_Y>rand(1));
            if mm_min(1)==2
                Y{n}(t)=1;
            else
                Y{n}(t)=0;
            end
        end
    temp = max((R(1,1)*(1-phi1_s).*(1-Y{n}) + (1-R(1,1))*phi1_s.*(1-Y{n})...
        +(1-R(2,2))*(1-phi1_s).*Y{n}+R(2,2)*phi1_s.*Y{n}),myeps);
    %%temp1 = max(R(1,1)*(1-phi1_s) + (1-R(1,1))*phi1_s);
    %%temp2 = max((1-R(2,2))*(1-phi1_s)+R(2,2)*phi1_s,myeps);
    %%% add increment of function value with respect to this cohort
    %%LogL=LogL + sum(log(temp1).*(1-Y{n}))+sum(log(temp2).*Y{n});

    % add increment of function value with respect to this cohort
    LogL=LogL + sum(log(temp));

 end

LogL=-(1/(TT_full))*LogL;%+eps1*sum(abs(P(2:N_X)));
end

