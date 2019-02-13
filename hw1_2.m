%read image and change format into gray
img = imread('hw1-p1.png') 
img= rgb2gray(img);
img=single(img);     %the input of sift must be gray-scale image in single precision
[f1,d1]=vl_sift(img); %extract features by sift function 
img2= imread('hw1-p2.png'); 
img2= rgb2gray(img2);
img2=single(img2);
[f2,d2]=vl_sift(img2);

[matches,scores]=vl_ubcmatch(d1,d2);
numMatches=size(matches,2);

X1=f1(1:2,matches(1,:)) ; X1(3,:)=1 ;
X2=f2(1:2,matches(2,:)) ; X2(3,:)=1 ;

clear H score ok;
for t=1:100
    subset=vl_colsubset(1:numMatches,4);
    A=[];
    for i=subset
        A=cat(1,A,kron(X1(:,i)' , vl_hat(X2(:,i))));
    end
    [U,S,V]=svd(A) ;
    H{t}=reshape(V(:,9),3,3) ;
    
    %homography from image1 to image2
    X2_=H{t}*X1;
    du = X2_(1,:) ./X2_(3,:) - X2(1,:)./X2(3,:) ;
    dv=  X2_(2,:) ./X2_(3,:) - X2(2,:)./X2(3,:) ;
    ok{t}=(du.*du+dv.*dv)<6*6 ;
    score(t)=sum(ok{t});
end

[score,best]=max(score);
H=H{best}
ok=ok{best}
