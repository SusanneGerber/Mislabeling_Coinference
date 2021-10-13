function [Y,LogL,LogL_range] = GenerateTrajectoryGLM_with_Mislabeling_v2(x,X,GLM_function,r_range)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% the epsilon used for avoiding infinity
myeps = 1e-13;
N_X=size(X{1},2);
P=x(1:N_X);%tau=x(2*N_X);
N_cohorts=numel(X);
risk=x(2*N_X:2*N_X+1);
N_range=length(r_range);
TT_full=0;
for n=1:N_cohorts
    TT_full=TT_full+size(X{n},1);
end
LogL=0;LogL_range=zeros(1,N_range);
R=[1-risk(1) risk(2);risk(1) 1-risk(2)];
for ind_r=1:N_range
    rrr(:,:,ind_r)=[1-r_range(ind_r) risk(2);r_range(ind_r) 1-risk(2)];
end
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
%    temp = max((R(1,1)*(1-phi1_s).*(1-Y{n}) + (1-R(1,1))*phi1_s.*(1-Y{n})...
%        +(1-R(2,2))*(1-phi1_s).*Y{n}+R(2,2)*phi1_s.*Y{n}),myeps);
temp = max((R(1,1)*(1-phi1_s).*(1-Y{n}) + R(1,2)*phi1_s.*(1-Y{n})...
    +R(2,1)*(1-phi1_s).*Y{n}+R(2,2)*phi1_s.*Y{n}),myeps);
    temp_r=zeros(1,N_range);
    for ind_r=1:N_range
        temp_r(ind_r)=sum(log(max((rrr(1,1,ind_r)*(1-phi1_s).*(1-Y{n}) + rrr(1,2,ind_r)*phi1_s.*(1-Y{n})...
        +rrr(2,1,ind_r)*(1-phi1_s).*Y{n}+rrr(2,2,ind_r)*phi1_s.*Y{n}),myeps)));
    end
    %%temp1 = max(R(1,1)*(1-phi1_s) + (1-R(1,1))*phi1_s);
    %%temp2 = max((1-R(2,2))*(1-phi1_s)+R(2,2)*phi1_s,myeps);
    %%% add increment of function value with respect to this cohort
    %%LogL=LogL + sum(log(temp1).*(1-Y{n}))+sum(log(temp2).*Y{n});

    % add increment of function value with respect to this cohort
    LogL=LogL + sum(log(temp));
    LogL_range=LogL_range+temp_r; 
 end

LogL=-(1/(TT_full))*LogL;%+eps1*sum(abs(P(2:N_X)));
LogL_range=-(1/(TT_full))*LogL_range;
end

