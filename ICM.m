function segmentation=ICM(image,class_number,potential,maxIter)
% Returns matrix of segmented classes
% INPUT : image        --> matrix of double, it could be 2D or 3D image
%         class_number --> wanted number of segmented classes
%         potential    --> pairwise potential (Add smoothness between
%         pixels)
%         maxIter      --> Stop condition
[width,height,bands]=size(image);
image=imstack2vectors(image);
%[segmentation,c]=kmeans(image,class_number);

%Creation of a GMM to get model params,   
GMModel = fitgmdist(image,class_number,'Replicates',30,'CovarianceType','diagonal','RegularizationValue',1e-4,'Options',statset('MaxIter',1000));
segmentation = cluster(GMModel,image);

%segmentation=reshape(id,[width height]);

%Instead of using join probability, we can use the term of energy
%We get two terms of energy, the firs one is the energy of feature field
%or the unary energy (the sum of the unary terms which pays a penalty for disregarding the
%annotation), the second one corresponds to Label field energy, which
%reflects the smoothness between pixels.

%We performe iterative optimisation to get to the minimum of distribution
%energy
%We already have the GMM initialized
%At each iteration we start by learning the GMM params
%Calculate the energy of distribution
%Getting the minimum of energy
%We get the best of segmentation at the end of process
clear c;
iter=0;
while(iter<maxIter)
    [mu,sigma]=GMM_parameter(image,segmentation,class_number);
    Ef=EnergyOfFeatureField(image,mu,sigma,class_number);
    E1=EnergyOfLabelField(segmentation,potential,width,height,class_number);
    E=Ef+E1;
    [tm,segmentation]=min(E,[],2);
    iter=iter+1;
end
%segmentation=reshape(segmentation,[width height]);
end