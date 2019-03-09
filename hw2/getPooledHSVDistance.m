function [dist_matrix]=getPooledHSVDistance(hist1, hist2)
d1=mean(min(pdist2(hist1, hist2)));  %we can get a spercific distance number rather than a matrix 
d2=mean(min(pdist2(hist2, hist1))); % we select each columns min value and then get average value 
dist_matrix=min(d1,d2);    
end

