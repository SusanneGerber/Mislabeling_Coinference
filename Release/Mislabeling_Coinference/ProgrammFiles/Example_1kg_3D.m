clear all
close all

load('Wellderly.mat');
load('ind_sort');
XX=XX(:,ind_sort(1:60));label=label(ind_sort(1:60));
YY=YY';ZZ=ZZ';
ind_malignant=find(YY==1);%
ind_benign=setdiff(1:length(YY),ind_malignant);

X{1}=XX(ind_malignant,:);X{2}=XX(ind_benign,:);
Y{1}=YY(ind_malignant,:);Y{2}=YY(ind_benign,:);
Z{1}=ZZ(ind_malignant,:);Z{2}=ZZ(ind_benign,:);
T{1}=length(ind_malignant);T{2}=length(ind_benign);
poolobj = parpool(12);
mislabeling_1kg=[0.0:0.02:0.3];
C=[150:-5:30];%[5e4 1e4 5000 1000 100 50 30 20 18 15 10 5];
ParallelBlockSize=200;
GLM_function='logit';%'linear';%
N_cohorts=numel(T);
train_proportion=75/100;%9/10;%2/3;
N_bootstrap=100;
N_X=size(X{1},2);
risk_cohort_visualise=2;
risk{2}=[0:0.02:0.3];%[0 0.001 0.001*2^2 0.001*2^3 0.001*2^4 0.001*2^5 0.001*2^6 0.001*2^7];

P=zeros(length(C),2*N_X-1+N_cohorts,length(risk{risk_cohort_visualise}),N_bootstrap,length(mislabeling_1kg));

for ind_1kg=12:length(mislabeling_1kg)
    ind_1kg
    risk{1}=mislabeling_1kg(ind_1kg);
    randn('seed',1);
    rand('seed',1);
     for n_bootstrap=1:N_bootstrap
        n_bootstrap
        for n=1:N_cohorts
            TT=T{n};
            ind{n}=randperm(TT);%Y=Y(:,[2 1]);
            [mm,ind_back{n}]=sort(ind{n});
            X{n}=X{n}(ind{n},:);Y{n}=Y{n}(ind{n},:);Z{n}=Z{n}(ind{n},:);
            N_train{n}=round(train_proportion*TT);
            X_train{n}=X{n}(1:N_train{n},:);Y_train{n}=Y{n}(1:N_train{n},:);Z_train{n}=Z{n}(1:N_train{n},:);
            X_valid{n}=X{n}((1+N_train{n}):TT,:);Y_valid{n}=Y{n}((1+N_train{n}):TT,:);Z_valid{n}=Z{n}((1+N_train{n}):TT,:);
            TT_valid{n}=size(X_valid{n},1);
        end
        
        
        NST=2;
        MEM=1;
        
        i_s=1;i_ns=1;
        P_init=[];
        ind_init=[];
        clear LogL BIC ind Pl LogLl BICl indl
        % for n_eps=1:length(EPS)
        %     n_eps
        %     [Pl(n_eps,:),LogLl(n_eps),BICl(n_eps),indl] = DiscreteCorrelationMultiCategorial_v3_Logit(YY,XX',P_init,EPS(n_eps));
        %     P_init=Pl(n_eps,:);
        %     %ind_init=ind;
        % end
        
        tt_ind=1;
        for i=1:length(C)
            %disp(['i=' num2str(i) '; regularization constant C(i)=' num2str(C(i))]);
            if n_bootstrap==1
                if tt_ind==1
                    in.xxx_init=[];
                else
                    in.xxx_init=outs{tt_ind-1}.P;
                end
            else
                in.xxx_init=squeeze(P(i,:,1,n_bootstrap-1));
            end
            in.Y=Y_train;
            in.X=X_train;
            in.TT=N_train;
            in.GLM_function=GLM_function;
            in.eps1=C(tt_ind);
            in.risk=risk;
            in.risk{2}=risk{2}(1);
            in.anneal=2;
            in.Y_valid=Y_valid;
            in.X_valid=X_valid;
            in.TT_valid=TT_valid;
            if n_bootstrap==1
                outs{tt_ind} = DiscreteGLM_cohort(in);
                
                %outs{tt_ind}.P=
                LogL(tt_ind)=outs{tt_ind}.LogL;
                LogL_valid(tt_ind)=outs{tt_ind}.LogL_valid;
                AICc(tt_ind)=outs{tt_ind}.AICc;
                BIC(tt_ind)=outs{tt_ind}.BIC;
            end
            tt_ind=tt_ind+1;
        end
        %figure;subplot(3,1,1);plot(C,-LogL_valid);subplot(3,1,2);plot(C,AICc,'r');
        %subplot(3,1,3);plot(C,BIC,'g');
        %[~,ii]=min(LogL_valid);
        %figure;plot(outs{ii}.P(1:N_X),'k-o');
        %    title(['fff=' num2str(outs{ii}.LogL_valid) ' ; risk=' num2str(outs{ii}.P(2*N_X:2*N_X-1+N_cohorts))]);
        
        
        tt=1;
        Block=1;
        clear in
        for ind_dr=1:length(risk{risk_cohort_visualise})
            for n_eps=1:length(C)
                if n_bootstrap==1
                    in{tt}.xxx_init=outs{n_eps}.P;
                    in{tt}.xxx_init(2*N_X-1+risk_cohort_visualise)=risk{risk_cohort_visualise}(ind_dr);
                else
                    in{tt}.xxx_init=P(n_eps,:,ind_dr,n_bootstrap-1);
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                in{tt}.Y=Y_train;
                in{tt}.X=X_train;
                in{tt}.TT=N_train;
                in{tt}.GLM_function=GLM_function;
                in{tt}.eps1=C(n_eps);
                for n=1:N_cohorts
                    if n~=risk_cohort_visualise
                        in{tt}.risk{n}=risk{n}(1);
                    else
                        in{tt}.risk{n}=risk{n}(ind_dr);
                    end
                end
                in{tt}.anneal=1;
                in{tt}.Y_valid=Y_valid;
                in{tt}.X_valid=X_valid;
                in{tt}.TT_valid=TT_valid;
                in{tt}.i=ind_dr;in{tt}.j=n_eps;
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if and(tt<ParallelBlockSize,(Block-1)*ParallelBlockSize+tt<length(risk{risk_cohort_visualise})*length(C))
                    tt=tt+1;
                else
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    if Block==1
                        kkk=0;
                    else
                        kkk=numel(out);
                    end
                    parfor ind_par=1:numel(in)
                        out{kkk+ind_par} = DiscreteGLM_cohort(in{ind_par});
                        out{kkk+ind_par}.i=in{ind_par}.i;
                        out{kkk+ind_par}.j=in{ind_par}.j;
                        out{kkk+ind_par}.index=kkk+ind_par;
                    end
                    %stop(tt_c);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    tt=1;
                    clear in
                    Block=Block+1;
                end
            end
        end
        
        
        
        for ttt=1:numel(out)
            for ind_p=1:length(out{ttt}.P)
                P(out{ttt}.j,ind_p,out{ttt}.i,n_bootstrap,ind_1kg)=out{ttt}.P(ind_p);
            end
            BICs(out{ttt}.j,out{ttt}.i,n_bootstrap,ind_1kg)=out{ttt}.BIC;
            AICcs(out{ttt}.j,out{ttt}.i,n_bootstrap,ind_1kg)=out{ttt}.AICc;
            LogLs(out{ttt}.j,out{ttt}.i,n_bootstrap,ind_1kg)=out{ttt}.LogL;
            LogLs_valid(out{ttt}.j,out{ttt}.i,n_bootstrap,ind_1kg)=out{ttt}.LogL_valid;
            t(ttt)=out{ttt}.index;
            r_fin(out{ttt}.j,out{ttt}.i,n_bootstrap,ind_1kg)=risk{risk_cohort_visualise}(out{ttt}.i);
            c_fin(out{ttt}.j,out{ttt}.i,n_bootstrap,ind_1kg)=C(out{ttt}.j);
        end
        clear out
    end
    save birads_intermediate_9_10
end
nx=2*size(X,2)-1;
[Cg,DRg]=meshgrid(C,risk{risk_cohort_visualise});
mean_AICc=zeros(size(AICcs,1),size(AICcs,2),size(AICcs,4));
mean_LogL_valid=zeros(size(AICcs,1),size(AICcs,2),size(AICcs,4));
mean_LogL=zeros(size(AICcs,1),size(AICcs,2),size(AICcs,4));
nnn=size(AICcs,3);
for n_bootstrap=1:nnn
    [mmm_AIC(n_bootstrap),i,j,l]=min_matrix3D(squeeze(AICcs(:,:,n_bootstrap,:)));
    [v,i,j,l]=min_matrix3D(squeeze(LogLs_valid(:,:,n_bootstrap,:)));
    P_opt(:,n_bootstrap)=squeeze(P(i,:,j,n_bootstrap,l));
    C_opt(n_bootstrap)=C(i);r_opt_well(n_bootstrap)=[risk{2}(j)];
    r_opt_1kg(n_bootstrap)=mislabeling_1kg(l);
    %[mmm_AIC0(n_bootstrap),i]=min_matrix(LogLs_valid(:,1,n_bootstrap)');
    %P_opt0(:,n_bootstrap)=squeeze(P(length(C),:,1,n_bootstrap));
    mean_AICc=mean_AICc+(1/nnn)*squeeze(AICcs(:,:,n_bootstrap,:));
    %mean_LogL_valid=mean_LogL_valid+(1/nnn)*LogLs_valid(:,:,n_bootstrap);
    mean_LogL_valid=mean_LogL_valid+(1/nnn)*squeeze(LogLs_valid(:,:,n_bootstrap,:));
end
 figure;histogram(P_opt(size(P_opt,1)+risk_cohort_visualise-2,:),5,'Normalization','pdf')   

figure;contourf(Cg,DRg,mean_LogL_valid',[0.44:0.0005:0.5]);set(gca,'XScale','log');[v,i,j]=min_matrix(mean_LogL_valid');
hold on;plot(C(j),risk{risk_cohort_visualise}(i),'ro');%xlim([1 15])
figure;contourf(Cg,DRg,mean_AICc',[650:1:750]);[v,i,j]=min_matrix(mean_AICc');set(gca,'XScale','log');
hold on;plot(C(j),risk{risk_cohort_visualise}(i),'ro');%xlim([1 15])
figure;plot(P_opt(1:N_X,1:nnn));
figure;plot(P_opt0(1:N_X,1:nnn));
mean_P=mean(P_opt(1:N_X,:)');
mean_risk=mean(P_opt((2*N_X):(2*N_X+1),:)')
mean_P0=mean(P_opt0(1:N_X,:)');
[~,tind]=sort(YY,'ascend');
XX=XX(tind,:);YY=YY(tind);ZZ=ZZ(tind);
phi=[];
y=[];z=[];
    for t=1:size(XX,1)
        [xx]=feval(GLM_function,mean_P,XX(t,:));
        phi=[phi xx];
        y=[y YY(t)];
        if ZZ(t)==4
        z=[z 0.3];
        elseif ZZ(t)==5
        z=[z 0.95];
        else
        z=[z 0];
        end
    end
figure;plot(y,'b');hold on;plot(phi,'r--.');
[A,Aci] = auc([y' phi'],0.05,'hanley')
[Az,Aciz] = auc([y' z'],0.05,'hanley')

mmm=min(min(mean_AICc));
for i=1:size(mean_AICc,1)
    for j=1:size(mean_AICc,2)
        weight_posterior(i,j)=exp(-0.5*(mean_AICc(i,j)-mmm));
    end
end
d_risk=diff(risk{risk_cohort_visualise});d_C=diff(C);
[v1,v2]=meshgrid(d_risk,d_C);
[N1,N2]=size(v1);
integral=sum(sum(weight_posterior(1:N1,1:N2).*v1));
weight_posterior=weight_posterior./integral;
figure;surfc(Cg,DRg,weight_posterior');xlim([1 60]);alpha(0.8)
shading interp
colormap hot
set(gca,'LineWidth',2,'FontSize',16)
box on
       prob_zero_hypothesis=sum(weight_posterior(1:N1,1)'.*d_C.*d_risk(1))
for j=1:N2
    Error_Distr(j)=0;
    for i=1:N1
        Error_Distr(j)=Error_Distr(j)+weight_posterior(i,j)*d_C(i);
    end
end

figure;plot(risk{risk_cohort_visualise}(1:N2),Error_Distr,'.-');%mesh(Cg,DRg,weight_posterior')

[x_pl,x_mi]=EmpConfIntArray(mean_P,P_opt(1:N_X,:)',0.95);
figure;errorbar(1:length(mean_P),mean_P,x_mi,x_pl)
clear Impact
for i=2:N_X
    i
    kkk=1;
    for n_anneal=1:1
        ppp0=mean_P';ppp0(i)=0;
        ppp1=mean_P';
        for n=1:N_cohorts
            for t=1:size(X{n},1)
                if X{n}(t,i)>0
                    [xx0]=feval(GLM_function,ppp0(1:N_X)',X{n}(t,:));
                    [xx1]=feval(GLM_function,ppp1(1:N_X)',X{n}(t,:));
                    Impact{i}(kkk)=xx1-xx0;
                    kkk=kkk+1;
                end
            end
        end
    end
    Impact_mean(i)=mean(Impact{i});
    Impact_conf_int(i)=1.96*std(Impact{i});
end
[mm,ii]=sort(abs(Impact_mean),'descend');
for i=1:N_X-1
    disp(['i=' num2str(ii(i)) ', average impact of ' label{ii(i)} ' on risk is ' ...
        num2str(Impact_mean(ii(i))) ' +/- '...
        num2str(Impact_conf_int(ii(i))) ';']);
end
figure;contourf(Cg,DRg,LogLs_valid(:,:,n_bootstrap)',100);[v,i,j]=min_matrix(LogLs_valid(:,:,n_bootstrap)');
hold on;plot(C(j),risk{risk_cohort_visualise}(i),'ro');
figure;contourf(Cg,DRg,AICcs(:,:,n_bootstrap)',100);
[v,i,j]=min_matrix(AICcs(:,:,n_bootstrap)');
hold on;plot(C(j),risk{risk_cohort_visualise}(i),'ro');
figure;plot(squeeze(P_opt(1:N_X,n_bootstrap)),'k-o');
title(['L(validation data)=' num2str(v) ' ; risk=' num2str(risk{risk_cohort_visualise}(i))]);
pause(0.2)


