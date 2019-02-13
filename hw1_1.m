load('data_batch_1.mat')
classNum=10;
images=data(1:1000,:); 
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
%we already had a histogram dataset :total_histImage as the dataset
%we need to select random data from dataset and let it compare distance with t(centroid)
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
random=randi([0 1000],100,1)'
for i =random
  randomImage=total_histImage2(i,:);
  randomDisSet = pdist2(randomImage,t);
  afterSorted=sort(randomDisSet)
  afterSorted(1);
  findResult=find(randomDisSet==afterSorted(1))
  fprintf("The category is %d",(findResult-1))
end

    


 

