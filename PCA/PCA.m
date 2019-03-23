%we can design our simple classifer for ficial recognation
%we need to use PCA to get important eigen vector and eigen values
%we already have those matrix, we do genuine and imposters caculation.
%genuine scores matrix come from same subject , we use pdist2() to get distance
%for one test image we have five genuine score ,
%for inposter, ew need to compare with different subjects randomly.
cat_list = dir('att_faces');
cat=40 , test_samples=5,train_samples=5
train=zeros(10304,200);
test=zeros(10304,200);
n=0
%read the top 5 images as traininig data from folder by name 
for i=1:cat
    dirName=cat_list(i+3).name;
    flist=dir(sprintf('att_faces/%s/*.pgm',dirName));
    for j=1:train_samples
         n=n+1;
         imtrain{n} = imread(sprintf('att_faces/%s/%s',dirName, flist(j).name));
    end
end
 %reshape the image 
 %and then we can get a matrix,from columns 1:5 it represent subject 1
 %->195:200 represents subject 40
 m=0
for i=1:cat
    dirName=cat_list(i+3).name;
    flist=dir(sprintf('att_faces/%s/*.pgm',dirName));
    for j=test_samples+1:test_samples+train_samples
         m=m+1;
         imtest{m} = imread(sprintf('att_faces/%s/%s',dirName, flist(j).name));
    end
end
for j=1:200
    m_test=cell2mat(imtest(j));
    m_train=cell2mat(imtrain(j));
    m_test=reshape(m_test,[112*92,1]);
    m_train=reshape(m_train,[112*92,1]);
    train(:,j)=m_train;
    test(:,j)=m_test;
end

% Compute the mean of the data matrix "The mean of each row"
meanTrain = mean(train')';
meanTest=mean(test')';
% Subtract the mean from each image [Centering the data]
%zero-mean
d=train-repmat(meanTrain,1,200);
dtest=test-repmat(meanTest,1,200);
%calculate the covaiance matrix
co=d*d';

%calculate the eigen values and eigen vectors of the covariance.
[eigvector,eigvl]=eig(co);

% Sort the eigen vectors according to the eigen values
%create diagonal matrix
eigvalue = diag(eigvl);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = eigvector(:, index);

% Compute the number of eigen values that greater than zero (you can select any threshold)
count1=0;
for i = 1:size(eigvalue,1)
    if eigvalue(i)>0
        count1=count1+1 ;  %means is important to our data
    end
end


%extract the top count1 eigen vector from eigvector set
vec=eigvector(:,1:count1);

% make projection to a subspace for test and train with more useful and
% powerful features
eTrain=vec'*d;
eTest=vec'*dtest;

imposter=[];
genuine=[];
%genuine is the distance for same subject (internal)
for j=1:cat
        train=eTrain(:,((j-1)*5+1):((j-1)*5+5));
    for i=1:train_samples
        test=eTest(:,(j-1)*5+i);
        gen=pdist2(test',train','Euclidean');
        genuine=[genuine;gen];
    end
end
%imposter is the distance for different subject randomly(external)
%one subject compares with only one different particular subject 
for j=1:cat
    ran=randi([1 40],1,1)
    if ran==j
        ran=randi([1 40],1,1);
    end
    train=eTrain(:,((ran-1)*5+1):((ran-1)*5+5));
    for i=1:train_samples
        test=eTest(:,(j-1)*5+i);
        impos=pdist2(test',train','Euclidean');
        imposter=[imposter;impos];
    end
end

%merege imposter and genuine //columns 1-5 are genuine 6-10 are imposter
scores=[genuine,imposter];
%normalize the score 
scores_nom=(scores-min(scores))/(max(scores)-min(scores))

%min(scores_nom(:)) max(scores_nom(:))
minScores=min(scores_nom(:))  %min=0.2118
maxScores=max(scores_nom(:))    %max=0.78

threshold=(minScores+maxScores)/2

ground_truth=zeros(200,10)
ground_truth(1:200,1:5)=0
ground_truth(1:200,6:10)=1

scores=scores/10000
predicted=zeros(200,10)
for i=1:200
    for j=1:10
       if scores(i,j)<threshold
           predicted(i,j)=0 %genuine
       else  predicted(i,j)=1     %imposter
       end
    end
end

%fPR(FAR) imposter->geniune 
FP=0
TN=0
for i=1:200
    for j=test_samples+1:10
        if predicted(i,j)==0
            FP=FP+1
        else 
            TN=TN+1
        end
    end
end
FN=0
TP=0
%FRR geniune->imposter
for i=1:200
    for j=1:train_samples
        if predicted(i,j)==0
            TP=TP+1
        else 
            FN=FN+1
        end
    end
end

FAR=FP/(FP+TN)
FRR=FN/(TP+FN)
%the accuracy we computed with  a specific threshold 
ACC=1-(FAR+FRR)/2

%now we can get a roc curve with a continous threshold 
%I add two returned value to get a FAR and FRR sets to see the result
[FARroc,FRRroc,roc,EER,area,EERthr,ALLthr,d,gen,imp]=ezroc3(scores,ground_truth,2,'value',1)
