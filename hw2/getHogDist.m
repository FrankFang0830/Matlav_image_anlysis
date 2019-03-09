function [hogDis]=getHogDist(h1, h2)
%  for i=1:2
%      for j=1:2
%          sizeTotal=size(h1{i,j})
%          size1=sizeTotal(1)
%          size2=sizeTotal(2)
%          size3=sizeTotal(3)
%          im1=reshape (h1{i,j},[size1*size2,size3])
%          im2=reshape(h2{i,j},[size1*size2,size3])
%          d1=mean(min(pdist2(im1, im2)));
%          d2=mean(min(pdist2(im2, im1))); 
%          %hodDis{1,2} reflect the hog distance between h1{1,2} and h2{1,2}
%          %merge four distance into one struct
%          hogDis{i,j}=min(d1,d2);
%      end
%  end
         sizeTotal=size(h1)
         size1=sizeTotal(1)
         size2=sizeTotal(2)
         size3=sizeTotal(3)
         im1=reshape (h1,[size1*size2,size3])
         im2=reshape (h2,[size1*size2,size3])
         d1=mean(min(pdist2(im1, im2)));
         d2=mean(min(pdist2(im2, im1))); 
         hogDis=min(d1,d2);
end

