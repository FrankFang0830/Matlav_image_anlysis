load('data_batch_1.mat')
classNum=10;
images=data(1:1000,:); 
summ=0
for j=0:9                   %use for-loop to visit each category by labels
   index=find(labels==j)     %we can obtain all the datas from one specific category by find()
   imageRep=zeros(10,3072);    %representatives ,for 1 class we select top 10 images to use kmeans to cluster
   for i=1:10
   imageRep(i,:)=images(index(i),:)
   end

%question1
    for i=1:10
    imageRep1=imageRep(i,:);
    imageRepRgb=reshape(imageRep1,[32,32,3]);
    imageRep2hsv(i,:,:,:)=rgb2hsv(imageRepRgb);
    end
    imageRep_hsv=reshape(imageRep2hsv,[10*32*32,3])
    %we use k-means which k=72 to cluster the images
    %k-means method:first:initilize  centroids by k training data randomly 
    %second:mulitiple iterations by inner loop to find more accurace centorids 
    %assign training samples to the nearst centroid.
    [category,center]=kmeans(imageRep_hsv,72)%the data has 72 entry 
    [a,~]=size(category);
    [b,~]=size(center);
    dis_c=pdist2(imageRep_hsv,center) %the dis_c is a set contains the distance between each images from the same category and center 
    [min_dist,pixel_bin] = min(dis_c');  % remove the inaccurate data 

    for k=1:b
        h(k) = length(find(pixel_bin==k));
    end
    t(j+1,1:b) = h/sum(h);            %t stores each category's centroid 
    
    figure(2);
    subplot(1,3,1);imshow(uint8(imageRepRgb)); grid on;
    subplot(1,3,2);bar(h);grid on;
    subplot(1,3,3);plot3(center(:,1),center(:,2),center(:,3),'*')
    xlabel('R');ylabel('G');zlabel('B')
end

%question2
total_histImage=[];     
for p=1:1000
    histImage=data(p,:);
    histImageRGB=reshape(histImage,[32,32,3]);
    histImageRGB2=rgb2hsv(histImageRGB);
    histImageHSV=reshape(histImageRGB2,[32*32,3]);
    
    histImage_1=hist(histImageHSV(:,1),24); %h 
    histImage_2=hist(histImageHSV(:,2),24); %s 
    histImage_3=hist(histImageHSV(:,3),24); %v
    histImage=[histImage_1,histImage_2,histImage_3];
    histImage=histImage/sum(histImage);
    
    total_histImage=[total_histImage;histImage];
end
 

% Question 3:
%compute the distance between each images(1000) and centroids of each category(t) 
distance = pdist2(total_histImage,t);

%question 4:
%we already had a histogram dataset :total_histImage2 as the dataset
%we need to select random datas from dataset and let it compare distance with t(centroid)
%put the distance into a list ,select the minimum one output its label by find function
total_histImage2=[];     
for p=4000:5000
    histImage=data(p,:);
    histImageRGB=reshape(histImage,[32,32,3]);
    histImageRGB2=rgb2hsv(histImageRGB);
    histImageHSV=reshape(histImageRGB2,[32*32,3]);
    
    histImage_1=hist(histImageHSV(:,1),24); %h 
    histImage_2=hist(histImageHSV(:,2),24); %s 
    histImage_3=hist(histImageHSV(:,3),24); %v
    histImage=[histImage_1,histImage_2,histImage_3];
    histImage=histImage/sum(histImage);
    
    total_histImage2=[total_histImage2;histImage];
end
random=randi([1 1000],100,1)'
final10=[];final1=[];final2=[];final3=[];final4=[];final5=[];final6=[];
final7=[];final8=[];final9=[];

for i =random
  randomImage=total_histImage2(i,:);
  randomDisSet = pdist2(randomImage,t);
  afterSorted=sort(randomDisSet)
  afterSorted(1);
  findResult=find(randomDisSet==afterSorted(1))
  fprintf("The category is %d",(findResult-1))
  if((findResult-1)~=labels(i))
   summ=summ+1;
  end
 %distance 
 if findResult==1
     final1=[final1,afterSorted(1)]
     
  elseif findResult==2
     final2=[final2,afterSorted(1)]
     
  elseif findResult==3
      final3=[final3,afterSorted(1)]
      
  elseif findResult==4
      final4=[final4,afterSorted(1)]
     
   elseif findResult==5
      final5=[final5,afterSorted(1)]
      
   elseif findResult==6
      final6=[final6,afterSorted(1)]
    
   elseif findResult==7
      final7=[final7,afterSorted(1)]
      
   elseif findResult==8
      final8=[final8,afterSorted(1)]
     
   elseif findResult==9
      final9=[final9,afterSorted(1)]
     
   elseif findResult==10
      final10=[final10,afterSorted(1)]
     
 end 
  
  
end
accuracy=0;
accuracy=(100-summ)/100;
fprintf("The category0")
sort(final1)
fprintf("The category1")
sort(final2)
fprintf("The category2")
sort(final3)
fprintf("The category3")
sort(final4)
fprintf("The category4")
sort(final5)
fprintf("The category5")
sort(final6)
fprintf("The category6")
sort(final7)
fprintf("The category7")
sort(final8)
fprintf("The category8")
sort(final9)
fprintf("The category9")
sort(final10)




