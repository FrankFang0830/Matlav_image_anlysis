%we load two datasets, then we can get two structers
loadlivTrain=load('featureMat_liv_train_bioLBP.mat')
loadlatexTrain=load('featureMat_Latex_train_bioLBP.mat')
%we need to use cat() to  obtain the tables in the structers
livTrain=cat(1,loadlivTrain.featureMat_liv_train_bioLBP)
%before we merage into one matrix, we need to label the data by live(1) 
livTrain(:,55)=1
latexTrain=cat(1,loadlatexTrain.featureMat_Latex_train_bioLBP)
%fake(0)
latexTrain(:,55)=0
%we can get a final matrix with label(category)
train=[livTrain;latexTrain]

%question1
%train the model 
md1=fitcnb(train(:,1:54),train(:,55))  %fitcnb(X,Y)
fprintf("According to the amount of data from latex and live the prior probability is:\n")
% ratio is 1:5  , we can get two prior: 0:16.7%   1:83.3%
fprintf("%s",md1.Prior) 

%qustion2
loadlivTest=load('featureMat_liv_test_bioLBP.mat')
loadlatexTest=load('featureMat_Latex_test_bioLBP.mat')
liveTest=cat(1,loadlivTest.featureMat_liv_test_bioLBP)
latexTest=cat(1,loadlatexTest.featureMat_Latex_test_bioLBP)
liveTest(:,55)=1
latexTest(:,55)=0
test=[liveTest;latexTest]
%predict(classifer,X)
labels=predict(md1,test(:,1:54)) %we only use the 1:54 columns to predict, clo55 is category

%question3
L1 = resubLoss(md1)
fprintf("Classification error by resubstitution %d",L1)
L2=loss(md1,train(:,1:54),train(:,55))
fprintf("Loss %d",L4)

%question4
loadGtest=load(('featureMat_Gelatine_test_bioLBP.mat'))
gTest=cat(1,loadGtest.featureMat_Gelatine_test_bioLBP)
gTest(:,55)=0
testAddG=[test;gTest]
labelG=predict(md1,testAddG(:,1:54))

%question5
L3 = resubLoss(md1)
fprintf("Classification error by resubstitution after added Gelatine %d",L3)
L4=loss(md1,train(:,1:54),train(:,55))
fprintf("Loss after added Gelatine %d",L4)


%we changed the test set the loss result still invariant
%the range of defination about loss is for the trainging set ,we dont
%change the training set.


%question6
md1.Prior=[0.6,0.4]
labels=predict(md1,test(:,1:54))
L5 = resubLoss(md1)
fprintf("Classification error by resubstitution after adjusted prior %d",L11)
L6=loss(md1,train(:,1:54),train(:,55))
fprintf("Loss after adjusted prior %d",L22)


