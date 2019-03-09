
% find folder with some info into a struct by path
cat_list = dir('tiny-imagenet-200/train');
n_cat = 10; n_img = 10; 
n = 0;
ids=zeros(10,9);
% load images one category by one gategory
for k=1:n_cat
    ids(k,:) = cat_list(k+3).name; %when k=1 k+2=3 cat_list will return to a invalid folder ;we can obtain folder1-10
    fprintf('\n cat %s', ids(k,:));  %print folder name from ids
    flist = dir(sprintf('tiny-imagenet-200/train/%s/images/*.JPEG',ids(k,:)));
    for j=1:n_img
        n=n+1; %after k loop, n=100
        ims{n} = imread(sprintf('tiny-imagenet-200/train/%s/images/%s',ids(k,:), flist(j).name));
        fprintf('.');
    end
end
% associated labels
ids = kron([1:n_cat], ones(1,n_img))';  %ids[] is the labels of category. 

 
%plot
for k=1:100
    figure(11); 
    subplot(10,10,k); imshow(ims{k}); title(sprintf('cat:%d', ids(k)));
end

%question1
%k-means to get centriods
%we need to transfer ims from jpg to hsv
imshsv=[]
for k=1:(n_cat*n_img)
    c=cell2mat(ims(k))  %tranfer from cell to matrix 
    cc=reshape(c,[64,64,3])
    imshsv(k,:,:,:)=rgb2hsv(cc)
end
imshsv1=reshape(imshsv,[(n_cat*n_img)*64*64,3])
%we put all images from hsv format into kmeans to get the centroids
[~, centers] = kmeans(imshsv1,64)
%calculate histogram for each image
hh=getPooledHSVHistogram(cc, centers, [2,2])
hog1=getImHog(cc,[2,2])
c1=cell2mat(ims(98))  %tranfer from cell to matrix 
cc1=reshape(c1,[64,64,3])
hh1=getPooledHSVHistogram(cc1, centers, [2,2])
hog2=getImHog(cc1,[2,2])
hogdist=getHogDist(hog1,hog2)


load hw2-A-gmm-km.mat;
d_all = [];
for i = 1:100
    im = ims(i);
    im_mat=cell2mat(ims(i)) 
    im_gray_single = single(rgb2gray(im_mat));
    
    %VL_DSIFT() does NOT compute a Gaussian scale space of the image
    %we need to smooth the data before we make it.
    h0 = fspecial('gaussian', 3, 1.5);
    % convolution 
    im0 = imfilter(im_gray_single, h0);
    % sift
    [~, sift2] = vl_dsift(im0, 'step', 2, 'size', 3);%sift2 is a 128 x NUMKEYPOINTS matrix with one descriptor per column
    d = getSiftFv(sift2, A, gmm);% obtain by hw2-A-gmm-km
    d_all = [d_all, d];
end
distSIFT1=getSiftFvDis(d_all(:,99),d_all(:,100))

% hsv distance
hsvDis=zeros(100,100);
for i=1:100
    im1=ims{i}
    h1=getPooledHSVHistogram(im1,centers,[2,2])
    hsvDisByRow=zeros(1,100);
    for j=i:100
        im2=ims{j}
        h2=getPooledHSVHistogram(im2,centers,[2,2])
        dis=getPooledHSVDistance(h1,h2)
        hsvDisByRow(:,j)=dis
    end
    hsvDis(i,:)=hsvDisByRow
end
%hog distance
HogDis=zeros(100,100)
for i=1:100
    im1=ims{i}
    h1=getImHog(im1,[2,2])
    hogDisByRow=[]
    for j=i:100
        im2=ims{j}
        h2=getImHog(im2,[2,2])
        dis=getHogDist(h1,h2)
        hogDisByRow(1,j)=dis
    end
    HogDis(i,:)=hogDisByRow
end

%sift distance


for i=1:100
    for j=1:100
        SiftDis(i,j)=SiftDis(j,i);
    end
    
end
%ground dist
groundDis=pdist2(ids,ids)

%fusion
w1=1/1000; w2=1/5; w3=1/100;
Fuseddist= w3*hsvDis + HogDis + w2*SiftDis; 

%plot distance beween 
figure(4); 
subplot(3,4,1); imagesc(hsvDis); title('pooled histogram dist');
subplot(3,4,2); imagesc(HogDis); title('hog dist');
subplot(3,4,3); imagesc(SiftDis); title('dense sift fv dist');
subplot(3,4,4); imagesc(groundDis); title('ground dist');


% find images in same category by groundDis equals to zero.
% plot ROC
% elements in scores/labels((i-1)*10)+1):i*10 represents category:n 
% in my assumption each element compare with its corresponding by vl_roc
labels=ones(100,1)
scores1=zeros(100,1)
scores2=zeros(100,1)
scores3=zeros(100,1)
scores4=zeros(100,1)

for i=1:9
    scores1((((i-1)*10)+1):i*10,1)=hsvDis((((i-1)*10)+1),(((i-1)*10)+1):i*10);
    scores2((((i-1)*10)+1):i*10,1)=HogDis((((i-1)*10)+1),(((i-1)*10)+1):i*10);
    scores3((((i-1)*10)+1):i*10,1)=SiftDis((((i-1)*10)+1),(((i-1)*10)+1):i*10);
    scores4((((i-1)*10)+1):i*10,1)=SiftDis((((i-1)*10)+1),(((i-1)*10)+1):i*10);
end
subplot(3,4,9); hold on; grid on;  title('hist ROC');
vl_roc(labels, scores1);
subplot(3,4,10); hold on; grid on;  title('hog ROC');
vl_roc(labels, scores2);
subplot(3,4,11); hold on; grid on;  title('dense sift ROC');
vl_roc(labels, scores3);
subplot(3,4,12); hold on; grid on;  title('fused ROC');
vl_roc(labels, scores4);


subplot(3,4,5); hold on; grid on; title('hist dist');
[h1, v1]=hist(hsvDis(find(groundDis==0)), 64); plot(v1, h1, '.-r'); 
[h2, v2]=hist(hsvDis(find(groundDis>0)), 64); plot(v2, h2, '.-k'); 
subplot(3,4,6); hold on; grid on; title('hog dist');
[h1, v1]=hist(HogDis(find(groundDis==0)), 64); plot(v1, h1, '.-r'); 
[h2, v2]=hist(HogDis(find(groundDis>0)), 64); plot(v2, h2, '.-k'); 
subplot(3,4,7); hold on; grid on;  title('dense sift fv dist');
[h1, v1]=hist(SiftDis(find(groundDis==0)), 64); plot(v1, h1, '.-r'); 
[h2, v2]=hist(SiftDis(find(groundDis>0)), 64); plot(v2, h2, '.-k'); 
    
