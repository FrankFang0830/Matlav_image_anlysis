function [pcaTrain,pcaTest] = fun_pca(train,test)
meanTrain = mean(train')';
meanTest=mean(test')';
% Subtract the mean from each image [Centering the data]
%zero-mean
zm_train=train-repmat(meanTrain,1,200);
zm_test=test-repmat(meanTest,1,200);
%calculate the covaiance matrix
co=zm_train*zm_train';

%calculate the eigen values and eigen vectors of the covariance.
[eigvector,eigvl]=eig(co);

% Sort the eigen vectors according to the eigen values
%create diagonal matrix
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

% % Compute the number of eigen values that greater than zero (you can select any threshold)
% count1=0;
% for i = 1:size(eigvalue,1)
%     if eigvalue(i)>0
%         count1=count1+1 ;  %means is important to our data
%     end
% end


%extract the top count1 eigen vector from eigvector set
vec=eigvector(:,1:50);

% make projection to a subspace for test and train with more useful and
% powerful features
pcaTrain=vec'*train;
pcaTest=vec'*test;
end

