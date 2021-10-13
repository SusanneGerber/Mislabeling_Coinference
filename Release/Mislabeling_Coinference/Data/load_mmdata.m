function [XX,YY,ZZ,label] = load_mmdata(filename)
%LOAD_MMDATA load mammography data
%   [XX,YY,ZZ,label] = LOAD_MMDATA(filename)
%
%  INPUT:
%   filename - name of file with data 
%              see matlab LOAD function for more info about format
%
%  OUPUT:
%   XX - matrix of data, each row corresponds to one record
%   YY - labels of data (cohort attribution)
%   ZZ -
%   label - cell with text labels of columns of XX
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%


% load rough data
Data=load(filename);

% process loaded data
ZZ=Data(:,1);
XXX=Data(:,2:end-1);
YY=Data(:,end);
XX=zeros(size(YY,1),28); 
for t=1:size(XXX,1)
    XX(t,:)=[1 XXX(t,2)==1:4 XXX(t,3)==1:5 XXX(t,4)==1:4 ...
            1-(XXX(t,2)==1:4) 1-(XXX(t,3)==1:5) 1-(XXX(t,4)==1:4) XXX(t,1)./100];
end

% define labels
label = cell(1,size(XX,2));

label{1}='base risk factor';
label{2}='shape round';
label{3}='shape oval';
label{4}='shape lobular';
label{5}='shape irregular';
label{6}='margin circumscribed';
label{7}='margin microlobulated';
label{8}='margin obscured';
label{9}='margin ill-defined';
label{10}='margin spiculated';
label{11}='density high';
label{12}='density iso';
label{13}='density low';
label{14}='density fat-containing';

label{15}='shape not round';
label{16}='shape not oval';
label{17}='shape not lobular';
label{18}='shape not irregular';
label{19}='margin not circumscribed';
label{20}='margin not microlobulated';
label{21}='margin not obscured';
label{22}='margin not ill-defined';
label{23}='margin not spiculated';
label{24}='density not high';
label{25}='density not iso';
label{26}='density not low';
label{27}='density not fat-containing';
label{28}='age';

end