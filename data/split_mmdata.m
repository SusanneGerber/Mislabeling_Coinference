function [ X, Y, Z ] = split_mmdata( XX, YY, ZZ )
%SPIT_MMDATA split data into cells based on labels
%   [X,Y,Z] = SPLIT_MMDATA(XX,YY,ZZ)
%
%  INPUT:
%   XX - matrix of data, each row corresponds to one record
%   YY - labels of data (cohort attribution)
%   ZZ -
%
%  OUPUT:
%   X - X{1} = XX(YY==0,:), X{2} = XX(YY==1,:)
%   Y - Y{1} = YY(YY==0,:)=0, Y{2} = YY(YY==1,:)=1
%   Z - Z{1} = ZZ(YY==0,:), Z{2} = ZZ(YY==1,:)
%
% Gerber S., Pospisil L., Fournier D., Torkamani A., Rueda M., Horenko I.
% Published under MIT License, 2017-2018
%

ind_benign=find(YY==0);
ind_malignant=find(YY==1);
X{1}=XX(ind_malignant,:);X{2}=XX(ind_benign,:);
Y{1}=YY(ind_malignant,:);Y{2}=YY(ind_benign,:);
Z{1}=ZZ(ind_malignant,:);Z{2}=ZZ(ind_benign,:);

end

