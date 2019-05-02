function [LDAtrain,LDAtest] = fun_lda(train,test)
LDAtrainMain=[]
for i=1:40
    LDAtrainMain(:,i)=mean(train(:,(i-1)*5+1:i*5),2);
    ma_train(:,(i-1)*5+1:i*5)=repmat(LDAtrainMain(:,i),1,5);
end
% Average of the mean of all classes
m=mean(train,2);

% Calculate the within class variance (SW)
sw=zeros(10304,10304);
for i=1:40
    s=train(:,(i-1)*5+1:i*5)-ma_train(:,(i-1)*5+1:i*5);
    si=s*s';
    sw=sw+si;
end
invsw=pinv(sw);
% if more than 2 classes calculate between class variance (SB)
sb=zeros(10304,10304);
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
% count1=0;
% for i = 1:size(eigvalue,1)
%     if eigvalue(i)>0   
%         count1=count1+1 ;  
%     end
% end
LDAvec=eigvector(:,1:36);
% 
%%lda projection
LDAtest=LDAvec'*test;
LDAtrain=LDAvec'*train;
end


