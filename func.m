function [err]=func(data,input)
if length(input)==0
    err=1;
else
k=10;
group=data(:,end);
class=unique(data(:,end));
for i=1:length(class)
    sa=[];
    sa=data((group==class(i)),:);
    [number_of_smile_samples,~] = size(sa); % Column-observation
    smile_subsample_segments1 = round(linspace(1,number_of_smile_samples,k+1)); % indices of subsample segmentation points    
    data_group{i}=sa;
    smile_subsample_segments{i}=smile_subsample_segments1;
end
Fit_temp=zeros(1,10);
for i=1:k
    data_ts=[];
    data_tr =[];
    for j=1:length(class)
        smile_subsample_segments1=smile_subsample_segments{j};
        sa=data_group{j};
        test= sa(smile_subsample_segments1(i):smile_subsample_segments1(i+1) , :); % current_test_smiles
        data_ts=[test;data_ts] ; %训练数据
        train = sa;
        train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
        data_tr =[train;data_tr];%训练数据
    end
    mdl = fitcknn(data_tr(:,input),data_tr(:,end),'NumNeighbors',4,'Standardize',1);%训练KNN
    Ac1=predict(mdl,data_ts(:,input)); 
    Fit_temp(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
end
err=mean(Fit_temp);
end
end