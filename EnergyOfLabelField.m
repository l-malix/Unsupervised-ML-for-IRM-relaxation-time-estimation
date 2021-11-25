function [E]=EnergyOfLabelField(segmentation,potential,width,height,class_number)
n=size(segmentation,1);
segmentation=reshape(segmentation,[width height]);
NX=NeiX(segmentation);
Nei8=imstack2vectors(NX);
E=zeros(n,class_number);
for i=1:class_number
    E(:,i)=sum(Nei8~=i,2);
end
E=E*potential;
end