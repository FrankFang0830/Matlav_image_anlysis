function [h] = getPooledHSVHistogram(im, codebook, pooling)
   im=rgb2hsv(im)
   p1=pooling(1) %rows
   p2=pooling(2) %columns
   size1=64/p1   
   size2=64/p2
   %seprate into four different areas 
   %then reshape them to calculate distance
   part1=im(1:size1,1:size2,1:3)
   part2=im((size1+1:64),1:size2,1:3)
   part3=im(1:size1,(size2+1):64,1:3)
   part4=im((size1+1):64,(size2+1):64,1:3)
   part1=reshape(part1,[32*32,3])
   part2=reshape(part2,[32*32,3])
   part3=reshape(part3,[32*32,3])
   part4=reshape(part4,[32*32,3])
   %we obtain histogram by distance to each centriods 
   h1=pdist2(part1,codebook)
   h2=pdist2(part2,codebook)
   h3=pdist2(part3,codebook)
   h4=pdist2(part4,codebook)
   % remove the inaccurate data 
   [~,pixel_bin1] = min(h1');  
   [~,pixel_bin2] = min(h2');
   [~,pixel_bin3] = min(h3');
   [~,pixel_bin4] = min(h4');
   for k=1:64
        h1(k) = length(find(pixel_bin1==k));
        h2(k) = length(find(pixel_bin2==k));
        h3(k) = length(find(pixel_bin3==k));
        h4(k) = length(find(pixel_bin4==k));
   end
   h1=h1/sum(h1)
   h2=h2/sum(h2)
   h3=h3/sum(h3)
   h4=h4/sum(h4)
   h=[h1,h2,h3,h4];
end

