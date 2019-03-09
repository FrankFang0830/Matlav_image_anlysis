function [hogTotal]=getImHog(im,pooling)
   p1=pooling(1)
   p2=pooling(2)
   sizeIm=size(im)
   size1=sizeIm(1)/pooling(1)
   size2=sizeIm(2)/ pooling(2)
   cellSize=8
   %Divide image into 2x2 regions.
   part1=im(1:size1,1:size2,1:3)
   part2=im((size1+1:64),1:size2,1:3)
   part3=im(1:size1,(size2+1):64,1:3)
   part4=im((size1+1):64,(size2+1):64,1:3)
   part1=rgb2gray(part1)
   part2=rgb2gray(part2)
   part3=rgb2gray(part3)
   part4=rgb2gray(part4)
   sin1=im2single(part1)
   sin2=im2single(part2)
   sin3=im2single(part3)
   sin4=im2single(part4)
   hog1=vl_hog(sin1, cellSize, 'verbose', 'variant', 'dalaltriggs');
   hog2= vl_hog(sin2, cellSize, 'verbose', 'Variant','DalalTriggs') 
   hog3= vl_hog(sin3, cellSize, 'verbose', 'Variant','DalalTriggs') 
   hog4= vl_hog(sin4, cellSize, 'verbose', 'Variant','DalalTriggs') 
   hogTotal=[hog1,hog2,hog3,hog4]
   
end





