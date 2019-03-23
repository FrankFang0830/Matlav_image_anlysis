%we can design our simple classifer for ficial recognation
%we need to use PCA to get important eigen vector and eigen values
%we already have those matrix, we do genuine and imposters caculation.
%genuine scores matrix come from same subject , we use pdist2() to get distance
%for one test image we have five genuine score ,
%for inposter, ew need to compare with different subjects randomly.
cat_list = dir('att_faces');
cat=40 , test_samples=10,train_samples=10
train=zeros(10304,250);
test=zeros(10304,150);
n=0
%read the top 25 catergoies with 10  as traininig data from folder by name 
for i=1:25
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
for i=26:40
    dirName=cat_list(i+3).name;
    flist=dir(sprintf('att_faces/%s/*.pgm',dirName));
    for j=1:test_samples
         m=m+1;
         imtest{m} = imread(sprintf('att_faces/%s/%s',dirName, flist(j).name));
    end
end
for j=1:150
    m_test=cell2mat(imtest(j));
    m_test=reshape(m_test,[112*92,1]);
    test(:,j)=m_test;
end
for j=1:250
    m_train=cell2mat(imtrain(j));
    m_train=reshape(m_train,[112*92,1]);
    train(:,j)=m_train;
end
% Compute the mean of the data matrix "The mean of each row"
meanTrain = mean(train')';
meanTest=mean(test')';
% Subtract the mean from each image [Centering the data]
%zero-mean
d=train-repmat(meanTrain,1,250);
dtest=test-repmat(meanTest,1,150);
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
%we project the 15 test datasets into pca subspace
eTest=vec'*dtest;
 
imposter=[];
genuine=[];
%then we need to divide first five into test and remaining into training
%genuine is the distance for same subject (internal)
for j=1:15
        train=eTest(:,((j-1)*10+1):((j-1)*10+5));
    for i=6:10
        test=eTest(:,(j-1)*10+i);
        gen=pdist2(test',train','Euclidean');
        genuine=[genuine;gen];
    end
end

for j=1:15
    ran=randi([1 15],1,1)
    if ran==j
        ran=randi([1 15],1,1);
    end
    train=eTest(:,((ran-1)*10+1):((ran-1)*10+5));
    for i=1:5
        test=eTest(:,(j-1)*10+i);
        impos=pdist2(test',train','Euclidean');
        imposter=[imposter;impos];
    end
end
scores=[genuine,imposter];
ground_truth=zeros(75,10)
ground_truth(1:75,1:5)=0
ground_truth(1:75,6:10)=1

[FARroc,FRRroc,roc,EER,area,EERthr,ALLthr,d,gen,imp]=ezroc3(scores,ground_truth,2,'value',1)