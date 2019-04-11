%First we need to obtain eigen value and eigen vector by PCA
%based on eigen face we can reduce dimensionality 
%we project our data into PCA subspace
%do LDA
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
%reshape the image and put one image into one column
for j=1:200
    m_train=cell2mat(imtrain(j));
    m_train1=reshape(m_train,[112*92,1]);
    train(:,j)=m_train1;
end
% Compute the mean of the data matrix "The mean of each row"
meanTrain = mean(train')';
% Subtract the mean from each image [Centering the data]
%zero-mean
d=train-repmat(meanTrain,1,200);
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
count1=0;
for i = 1:size(eigvalue,1)
    if eigvalue(i)>0   
        count1=count1+1 ;  %means is important to our data
    end
end
vec=eigvector(:,1:count1);


%load test data
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
    m_test1=reshape(m_test,[112*92,1]);
    test(:,j)=m_test1;
end
meanTest = mean(test')';
dd=test-repmat(meanTest,1,200);
%project data into PCA subspace
PCAtrain=vec'*d;
PCAtest=vec'*dd;

LDAtrainMain=[]
m_train=zeros(4856,40)
% Calculating Mean of Each Class
% assign each class mean into their subject
for i=1:40
    LDAtrainMain(:,i)=mean(PCAtrain(:,(i-1)*5+1:i*5),2);
    m_train(:,(i-1)*5+1:i*5)=repmat(LDAtrainMain(:,i),1,5);
end
% Average of the mean of all classes
m=mean(PCAtrain,2);
% Center the data (zero-mean)
LDAtrain=PCAtrain-m_train;
% Calculate the within class variance (SW)
sw=zeros(4856,4856);
for i=1:40
    s=PCAtrain(:,(i-1)*5+1:i*5)-m_train(:,(i-1)*5+1:i*5);
    si=s*s';
    sw=sw+si;
end
invsw=pinv(sw);
% if more than 2 classes calculate between class variance (SB)
sb=zeros(4856,4856);
for i=1:40
     sii=5*(LDAtrainMain(:,i)-m)*((LDAtrainMain(:,i)-m)');
     sb=sb+sii;            %%Calculating between scatter matrix%%
end
jw=invsw*sb;
 
% find eigne values and eigen vectors 
[evec,eval]=eig(jw);
eval=real(eval);
evec=real(evec);
eigvalue = diag(eval);
[junk, index] = sort(eigvalue,'descend');
eigvalue = eigvalue(index);
eigvector = evec(:, index);
count1=0;
for i = 1:size(eigvalue,1)
    if eigvalue(i)>0   
        count1=count1+1 ;  
    end
end
LDAvec=eigvector(:,1:count1);
% 
%%lda projection
LDAtest=LDAvec'*PCAtest;
LDAtrain=LDAvec'*PCAtrain;

%we can get LDA subspace and then build our scores by imposter and genuine 
imposter=[];
genuine=[];
%genuine is the distance for same subject (internal)
for j=1:cat
        trainset=LDAtrain(:,(j-1)*5+1:j*5);
    for i=1:5
        testset=LDAtest(:,(j-1)*5+i);
        gen=pdist2(trainset',testset','Euclidean');
        genuine=[genuine;gen'];
    end
end

%one subject compares with entire dataset except itself 
temp=[]
imposter=[]
 for j=1:cat
   for i=1:cat
        if j~=i
            trainset=LDAtrain(:,((i-1)*5+1):i*5);
            for m=1:train_samples
                testset=LDAtest(:,(j-1)*5+m);
                dist=pdist2(testset',trainset','Euclidean');
                temp=[temp,dist']  ;%5*5*39 for one subject compare with the others 
            end
      end
   end
 end
%imposter is the distance for different subject randomly(external)
for j=1:cat
    ran=randi([1 40],1,1)
    if ran==j
        ran=randi([1 40],1,1);
    end
    train=LDAtrain(:,((ran-1)*5+1):((ran-1)*5+5));
    for i=1:train_samples
        test=LDAtest(:,(j-1)*5+i);
        impos=pdist2(test',train','Euclidean');
        imposter=[imposter;impos];
    end
end
scores=[genuine',temp]
scores=scores/1000
 
 groundtruth=zeros(5,8000);
 groundtruth(:,1:200)=0;
 groundtruth(:,201:8000)=1;
 [FARroc,FRRroc,roc,EER,area,EERthr,ALLthr,d,gen,imp]=ezroc3(scores,groundtruth,2,'value',1)
 