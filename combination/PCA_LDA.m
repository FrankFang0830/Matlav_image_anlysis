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

%we do PCA
[pcatrain,pcatest]=fun_pca(train,test)
%we do LDA
[ldatrain,ldatest]=fun_lda(train,test)

%ground truth
groundtruth=zeros(5,8000);
groundtruth(:,1:200)=0;
groundtruth(:,201:8000)=1;
 
 %genuine
imposter=[];
pcagenuine=[];
ldagenuine=[];
%genuine is the distance for same subject (internal)
for j=1:cat
        pcatrainset=pcatrain(:,(j-1)*5+1:j*5);
    for i=1:5
        testset=pcatest(:,(j-1)*5+i);
        pcagen=pdist2(pcatrainset',testset');
        pcagenuine=[pcagenuine;pcagen'];
    end
end
for j=1:cat
        ldatrainset=ldatrain(:,(j-1)*5+1:j*5);
    for i=1:5
        testset1=ldatest(:,(j-1)*5+i);
        ldagen=pdist2(ldatrainset',testset1');
        ldagenuine=[ldagenuine;ldagen'];
    end
end

%imposter
ldaimposter=[]
pcaimposter=[]
 for j=1:cat
   for i=1:cat
        if j~=i
            trainset=ldatrain(:,((i-1)*5+1):i*5);
            for m=1:train_samples
                testset=ldatest(:,(j-1)*5+m);
                dist=pdist2(testset',trainset','Euclidean');
                ldaimposter=[ldaimposter,dist']  ;%5*5*39 for one subject compare with the others 
            end
      end
   end
 end
  for j=1:cat
   for i=1:cat
        if j~=i
            trainset=pcatrain(:,((i-1)*5+1):i*5);
            for m=1:train_samples
                testset=pcatest(:,(j-1)*5+m);
                dist=pdist2(testset',trainset','Euclidean');
                pcaimposter=[pcaimposter,dist']  ;%5*5*39 for one subject compare with the others 
            end
      end
   end
  end
 
pcascores=[pcagenuine',pcaimposter];
ldascores=[ldagenuine',ldaimposter];
%we made a decision at the scores level
 
for i=1:8000
         tempmat=max(ldascores(:,i),pcascores(:,i));
         max_score(:,i)=tempmat;%Max score rule
         tempmat1=min(ldascores(:,i),pcascores(:,i));
         min_score(:,i)=tempmat1;%Min score rule
end;
for i=1:8000
    average_score(:,i)=(ldascores(:,i)+pcascores(:,i))/2;
end
max_score=max_score/1000;
min_score=min_score/1000;
pcascores=pcascores/1000;
ldascores=ldascores/1000;
average_score=average_score/1000;
[pcaFAR,pcaFRR,pcaroc,~,~,~,pcaALLthr,~,~,~]=ezroc3(pcascores,groundtruth,2,'value',1);
[ldaFAR,ldaFRR,ldaroc,~,~,~,ldaALLthr,~,~,~]=ezroc3(ldascores,groundtruth,2,'value',1);
[~,~,maxroc,~,~,~,~,~,~,~]=ezroc3(max_score,groundtruth,2,'value',1);
[~,~,minroc,~,~,~,~,~,~,~]=ezroc3(min_score,groundtruth,2,'value',1);
[~,~,avgroc,~,~,~,~,~,~,~]=ezroc3(average_score,groundtruth,2,'value',1);

figure(22), plot(maxroc(2,:),maxroc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]);
hold on
plot(minroc(2,:),minroc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]); hold on;
 
 plot(avgroc(2,:),avgroc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]); hold on;
 
 plot(pcaroc(2,:),pcaroc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]); hold on;
 
 plot(ldaroc(2,:),ldaroc(1,:),'LineWidth',3),axis([-0.002 1 0 1.002]); hold on;
 
 legend('max rule','min rule','average rule','pca','lda');
 

%EXPERIMENT2
pcalo1=mean(find(pcaFAR==0.005));
ldalo1=find(ldaFAR==0.005);
ldafRR=ldaFRR(ldalo1);
pcafRR=pcaFRR(pcalo1);

pcaACC=1-(0.005+pcafRR)/2;
ldaACC=1-(0.005+ldafRR)/2;
pcathr=pcaALLthr(pcalo1);
ldathr=ldaALLthr(ldalo1);

%combine PCA and LDA in the decision rule
pcapredicted=zeros(5,8000);
for i=1:5
    for j=1:8000
       if pcascores(i,j)<pcathr
           pcapredicted(i,j)=0 ;
       else  pcapredicted(i,j)=1   ; 
       end
    end
end
ldapredicted=zeros(5,8000);
for i=1:5
    for j=1:8000
       if ldascores(i,j)<ldathr
           ldapredicted(i,j)=0 ;
       else  ldapredicted(i,j)=1   ; 
       end
    end
end
finalpredicted=zeros(5,8000);
for i=1:5
    for j=1:8000
        if ldapredicted(i,j)==pcapredicted(i,j) && ldapredicted(i,j)==0
            finalpredicted(i,j)=0;
        elseif ldapredicted(i,j)==pcapredicted(i,j) && ldapredicted(i,j)==1
            finalpredicted(i,j)=1;
        else
            finalpredicted(i,j)=randi([0,1]);
        end
    end
end
TP=0
FN=0
for i=1:5
    for j=1:200
        if finalpredicted(i,j)==0
            TP=TP+1;
        else
            FN=FN+1;
        end
    end
end
FP=0
TN=0
for i=1:5
    for j=201:8000
        if finalpredicted(i,j)==1
            TN=TN+1;
        else
            FP=FP+1;
        end
    end
end
FAR=FP/(FP+TN);
FRR=FN/(TP+FN);
%the accuracy we computed with  a specific threshold 
ACC=1-(FAR+FRR)/2;




