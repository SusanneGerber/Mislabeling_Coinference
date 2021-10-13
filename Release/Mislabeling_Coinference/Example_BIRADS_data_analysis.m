%% Clear Values from last run 
clear all 
close all

%% add programm and data path
addpath('./Data/')
addpath('./ProgrammFiles/')
addpath('./Output/')


%% Prepare Dataset
Data = load('mammographic_masses.data.txt');

ZZ=Data(:,1);                  % first column of DF?                     
XXX=Data(:,2:size(Data,2)-1);  % actual data dispite first column and labels
YY=Data(:,size(Data,2));       % labels

for t=1:size(XXX,1)
XX(t,:)=[1 XXX(t,2)==1:4 XXX(t,3)==1:5 XXX(t,4)==1:4 ...
    1-(XXX(t,2)==1:4) 1-(XXX(t,3)==1:5) 1-(XXX(t,4)==1:4) XXX(t,1)./100];
end
%%

%% Labels for mamomgraphy Dataset
label{1}='base risk factor';
label{2}='shape round';label{3}='shape oval';
label{4}='shape lobular';label{5}='shape irregular';
label{6}='margin circumscribed';label{7}='margin microlobulated';
label{8}='margin obscured';label{9}='margin ill-defined';
label{10}='margin spiculated';label{11}='density high';label{12}='density iso';
label{13}='density low';label{14}='density fat-containing';

label{15}='shape not round';label{16}='shape not oval';
label{17}='shape not lobular';label{18}='shape not irregular';
label{19}='margin not circumscribed';label{20}='margin not microlobulated';
label{21}='margin not obscured';label{22}='margin not ill-defined';
label{23}='margin not spiculated';label{24}='density not high';label{25}='density not iso';
label{26}='density not low';label{27}='density not fat-containing';
label{28}='age';


%% Set labels for benign and malignant
ind_benign=find(YY==0);
ind_malignant=find(YY==1);


%% Preprocess data 
%split XX, YY and ZZ (frist column of data) by respective labels
X{1}=XX(ind_malignant,:);X{2}=XX(ind_benign,:);
Y{1}=YY(ind_malignant,:);Y{2}=YY(ind_benign,:);
Z{1}=ZZ(ind_malignant,:);Z{2}=ZZ(ind_benign,:);

% get number of malignant and benign samples
T{1}=length(ind_malignant);T{2}=length(ind_benign);

% Set parameters
risk{2}=0;
risk{1}=[0:0.005:0.05];
% set random seeds
randn('seed',1);
rand('seed',1);

risk_cohort_visualise=1;

% Set Model to logit 
GLM_function='logit';

% C is the a priori unknown constant that implicitly confines the number of
% non-zero components of the parameter vector α 
C=[1e5 120 60 20 15 14 13 12:-0.25:5 3 1];

% parallel comtutation block size
ParallelBlockSize=200;

%% Parameters
N_cohorts=numel(T);     % 2 for mammo 
train_proportion=3/4;   % train test split
N_bootstrap=2;          % number of bootstraps -> the more the better 
delete(gcp('nocreate'))
poolobj = parpool(7);  % start parpool with 7 workes (potentially 14 threads)

%% Computation

N_X=size(X{1},2); % Number of possible characteristics 
P=zeros(length(C),2*N_X-1+N_cohorts,length(risk{risk_cohort_visualise}),N_bootstrap); 
for n_bootstrap=1:N_bootstrap
    
    for n=1:N_cohorts
        TT=T{n};
        ind{n}=randperm(TT);
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
    tt_ind=1
    
    
    for i=1:length(C)
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
        in.risk{1}=risk{1}(1);
        in.anneal=3;
        in.Y_valid=Y_valid;
        in.X_valid=X_valid;
        in.TT_valid=TT_valid;
        if n_bootstrap==1
            
            
            outs{tt_ind} = DiscreteGLM_cohort(in);
            
            LogL(tt_ind)=outs{tt_ind}.LogL;
            LogL_valid(tt_ind)=outs{tt_ind}.LogL_valid;
            AICc(tt_ind)=outs{tt_ind}.AICc;
            BIC(tt_ind)=outs{tt_ind}.BIC;
        end
        tt_ind=tt_ind+1;
    end
    [~,ii]=min(LogL_valid);
    
    
    tt=1;
    Block=1
    clear in
    for ind_dr=1:length(risk{risk_cohort_visualise})
        for n_eps=1:length(C)
                 if n_bootstrap==1
                    in{tt}.xxx_init=outs{n_eps}.P;
                    in{tt}.xxx_init(2*N_X-1+risk_cohort_visualise)=risk{risk_cohort_visualise}(ind_dr);
                else
                    in{tt}.xxx_init=P(n_eps,:,ind_dr,n_bootstrap-1);
                end
           
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
            in{tt}.anneal=3;
            in{tt}.Y_valid=Y_valid;
            in{tt}.X_valid=X_valid;
            in{tt}.TT_valid=TT_valid;
            in{tt}.i=ind_dr;in{tt}.j=n_eps;
            
            
            if and(tt<ParallelBlockSize,(Block-1)*ParallelBlockSize+tt<length(risk{risk_cohort_visualise})*length(C))
                tt=tt+1;
            else
                
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
                 
                tt=1;
                clear in
                Block=Block+1
            end
        end
    end
    
    
    
    for ttt=1:numel(out)
        for ind_p=1:length(out{ttt}.P)
            P(out{ttt}.j,ind_p,out{ttt}.i,n_bootstrap)=out{ttt}.P(ind_p);
        end
        BICs(out{ttt}.j,out{ttt}.i,n_bootstrap)=out{ttt}.BIC;
        AICcs(out{ttt}.j,out{ttt}.i,n_bootstrap)=out{ttt}.AICc;
        LogLs(out{ttt}.j,out{ttt}.i,n_bootstrap)=out{ttt}.LogL;
        LogLs_valid(out{ttt}.j,out{ttt}.i,n_bootstrap)=out{ttt}.LogL_valid;
        t(ttt)=out{ttt}.index;
    end
    fprintf('Finished Bootstrap iter: %d\n ', n_bootstrap);
    save Output/birads_intermediate
    clear out
end

%% Plotting and statistics
clear P_opt P_opt0
nx=2*size(X,2)-1;

irl=11;
LogLs_valid=LogLs_valid(:,1:irl,:);
AICcs=AICcs(:,1:irl,:);
P=P(:,:,1:irl,1:size(AICcs,3));
risk{risk_cohort_visualise}=risk{risk_cohort_visualise}(1:irl);
[Cg,DRg]=meshgrid(C,risk{risk_cohort_visualise});
mean_AICc=zeros(size(AICcs,1),size(AICcs,2));
mean_LogL_valid=zeros(size(AICcs,1),size(AICcs,2));
mean_LogL=zeros(size(AICcs,1),size(AICcs,2));
nnn=size(AICcs,3);
for n_bootstrap=1:nnn
    [mmm_AIC(n_bootstrap),i,j]=min_matrix(AICcs(:,:,n_bootstrap)');
    %[v,i,j]=min_matrix(LogLs_valid(:,:,n_bootstrap)');
    P_opt(:,n_bootstrap)=squeeze(P(j,:,i,n_bootstrap));
    [mmm_AIC0(n_bootstrap),i]=min_matrix(LogLs_valid(:,1,n_bootstrap)');
    P_opt0(:,n_bootstrap)=squeeze(P(1,:,1,n_bootstrap));
    mean_AICc=mean_AICc+(1/nnn)*AICcs(:,:,n_bootstrap);
    mean_LogL_valid=mean_LogL_valid+(1/nnn)*LogLs_valid(:,:,n_bootstrap);
end
 figure;histogram(P_opt(size(P_opt,1)-1,:),8,'Normalization','pdf');%,'BinMethod','scott')   
 set(gca,'FontSize',16,'LineWidth',2);mm=mean(P_opt(size(P_opt,1)-1,:));
 hold on;plot(mm*[1 1],[0 200],'r--','LineWidth',2);
text(mm-0.0005,5,['Expected rate: ' num2str(mm)],'Rotation',90,'FontSize',16)
xlabel('Rate of false-positive breast biopsy outcomes');
ylabel('Posterior probability density function');

figure;contourf(Cg,DRg,mean_LogL_valid',[0.44:0.0001:0.47]);[v,i,j]=min_matrix(mean_LogL_valid');
hold on;plot(C(j),risk{risk_cohort_visualise}(i),'ro');xlim([4 20])
figure;contourf(Cg,DRg,mean_AICc',[540:0.25:590]);[v,i,j]=min_matrix(mean_AICc');
hold on;plot(C(j),risk{risk_cohort_visualise}(i),'ro');xlim([4 20])
figure;plot(P_opt(1:N_X,1:nnn));
figure;plot(P_opt0(1:N_X,1:nnn));
mean_P=mean(P_opt(1:N_X,:)');
mean_P0=mean(P_opt0(1:N_X,:)');
phi=[];
y=[];z=[];
for n=1:N_cohorts
    for t=1:T{n}
        [xx]=feval(GLM_function,mean_P0,X{n}(t,:));
        phi=[phi xx];
        y=[y Y{n}(t)];
        if Z{n}(t)==4
        z=[z 0.3];
        elseif Z{n}(t)==5
        z=[z 0.95];
        else
        z=[z 0];
        end
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
       prob_zero_hypothesis=sum(weight_posterior(1:N1,1)'.*d_C.*d_risk(1))
for j=1:N2
    Error_Distr(j)=0;
    for i=1:N1
        Error_Distr(j)=Error_Distr(j)+weight_posterior(i,j)*d_C(i);
    end
end

figure;plot(risk{risk_cohort_visualise}(1:N2),Error_Distr,'.-');%mesh(Cg,DRg,weight_posterior')

[x_pl,x_mi]=EmpConfIntArray(mean_P,P_opt',0.95);
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

nnn=size(AICcs,3);
AgeIntervals=[20:10:90];
for n_bootstrap=1:nnn
    for j=1:length(AgeIntervals)
    ppp0=zeros(1,N_X);ppp0(1)=P_opt(1,n_bootstrap);
    ppp0(N_X)=P_opt(N_X,n_bootstrap);
    X0=zeros(1,N_X);X0(1)=1;
    X0(N_X)=AgeIntervals(j)/100;
    [risk_age(n_bootstrap,j)]=feval(GLM_function,ppp0,X0);
    end
end
mean_risk_age=mean(risk_age);
[x_pl,x_mi]=EmpConfIntArray(mean_risk_age,risk_age,0.95);
figure;errorbar(AgeIntervals,mean_risk_age,x_mi,x_pl)





