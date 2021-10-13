function [ phi_mean,phi_conf,A_mean,A_conf] = ComputePersonalizedGLM_Risk(X,Y,param,risk,GLM_function)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

[N_ensemble,N_param]=size(param);
T=size(X,1);
phi=zeros(N_ensemble,T);

y=[];z=[];
if N_ensemble==1
    for t=1:T
        [xx]=feval(GLM_function,param,X(t,:));
        phi(t)=xx;
        y=[y Y(t)*(1-risk)];
    end
    phi_mean=phi;
    phi_conf=[0*phi; 0*phi];
    [A_mean,A_conf] = auc([y' phi'],0.05,'hanley');
else
    for n_ens=1:N_ensemble
        for t=1:T
            [xx]=feval(GLM_function,param(n_ens,:),X(t,:));
            phi(n_ens,t)=xx;
            if n_ens==1
                y=[y Y(t)*(1-risk)];
            end
        end
        [A(n_ens)] = auc([y' phi(n_ens,:)'],0.05,'hanley');
    end
    phi_mean=mean(phi);A_mean=mean(A);
    [phi_conf(1,:),phi_conf(2,:)]=EmpConfIntArray(phi_mean,phi,0.95);
    [A_conf(1),A_conf(2)]=EmpConfIntArray(A_mean,A,0.95);
end
phi_conf(1,:)=max(0.2*rand(1,length(phi_conf(1,:))),phi_conf(1,:));
phi_conf(2,:)=max(0.2*rand(1,length(phi_conf(2,:))),phi_conf(2,:));
for t=1:T
   d1= phi_mean(t)-phi_conf(2,t);
   if d1<0 
      phi_conf(2,t)= phi_mean(t);
   end
   d2= phi_mean(t)+phi_conf(2,t);
   if d2>1 
      phi_conf(1,t)= 1-phi_mean(t);
   end
end
end

