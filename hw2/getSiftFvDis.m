function [distSIFT] = getSiftFvDis(sift1,sift2)
     dis1=mean(min(pdist2(sift1,sift2)));
     dis2=mean(min(pdist2(sift2,sift1)));
     distSIFT=min(dis1,dis2)
end

