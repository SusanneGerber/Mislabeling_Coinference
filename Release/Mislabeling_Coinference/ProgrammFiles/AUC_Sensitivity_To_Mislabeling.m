function [ A_m,Az_m,Aci_m,Aciz_m ] = AUC_Sensitivity_To_Mislabeling(risk,X,Y,Z,P_model,GLM_function )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
N_ens=200;
N_cohorts=numel(X);
for n=1:N_cohorts
   T(n)=length(Y{n}); 
end
A_m=zeros(1,length(risk));
Az_m=zeros(1,length(risk));
Aci_m=zeros(2,length(risk));
Aciz_m=zeros(2,length(risk));
phi=[];
for n=1:N_cohorts
           for t=1:T(n)
                [xx]=feval(GLM_function,P_model,X{n}(t,:));
                phi=[phi xx];
           end
end

for ind_r=1:length(risk)
    ind_r
    R=[1-risk(ind_r) 0;risk(ind_r) 1];
    for ind_ens=1:N_ens
        y=[];z=[];
        for n=1:N_cohorts
            for t=1:T(n)
                pi=zeros(2,1);pi(Y{n}(t)+1)=1;
                pi_Y=cumsum(R*pi)';
                [~,mm_min]=find(pi_Y>rand(1));
                if mm_min(1)==2
                    Y_ens{n}(t)=1;
                else
                    Y_ens{n}(t)=0;
                end
            end
            for t=1:T(n)
                y=[y Y_ens{n}(t)];
                if Z{n}(t)==4
                    z=[z 0.3];
                elseif Z{n}(t)==5
                    z=[z 0.95];
                else
                    z=[z 0];
                end
            end
        end
        %figure;plot(y,'b');hold on;plot(phi,'r--.');
        [A(ind_ens,ind_r),Aci(:,ind_ens,ind_r)] = auc([y' phi'],0.05,'hanley');
        [Az(ind_ens,ind_r),Aciz(:,ind_ens,ind_r)] = auc([y' z'],0.05,'hanley');
         A_m(ind_r)=A_m(ind_r)+(1/N_ens)*A(ind_ens,ind_r);
         Az_m(ind_r)=Az_m(ind_r)+(1/N_ens)*Az(ind_ens,ind_r);
         Aci_m(:,ind_r)=Aci_m(:,ind_r)+(1/N_ens)*Aci(:,ind_ens,ind_r);
         Aciz_m(:,ind_r)=Aciz_m(:,ind_r)+(1/N_ens)*Aciz(:,ind_ens,ind_r);
    end
end

end

