clear all, close all force
addpath fonctions
addpath data
warning('off'); % pour désactiver l'affichage des warning de Matlab
%% Chargement des données

path=['./data/tomate_antenne_tete_2018/serie3_module_1_acc/'];
Image = getImagefromDicom(path);
axlim = [120 120 700 500 1600 1200];

% Calcul de la dimension de l'image
[xdim,ydim,tdim]=size(Image);

params.xdim=xdim;
params.ydim=ydim;
params.tdim=tdim;

params.FirstEcho = 6.5; %Temps du premier echo en ms
params.DeltaEcho = 6.5; %Temps d'echos entre deux echantillon en ms
params.EchoTime = (0:params.tdim-1)*params.DeltaEcho+params.FirstEcho;

%% Masquer le back-ground

img_forground = Image;
for i=1:tdim
    img = img_forground(:,:,i);
    seuil = graythresh(img)*255;
    img_binaire = img > seuil;
    img_forground(:,:,i) = img_binaire.*img;
end

figure;
imshow(img_forground(:,:,15),[]);
title('image without back-ground');
%% PCA pour la réduction de la dimension

%image_cropped = img_forground;%(26:105,26:95,:);

image_cropped = Image(26:105,26:95,:);
dimx=80;
dimy=70;


mask = mymask(image_cropped,0.15);

image_flat = zeros(length(mask),512);
for i=1:length(mask)
    [xi,yi] = myunravel(mask(i),dimx);
    image_flat(i,:) = image_cropped(xi, yi,:);
end

nbComponents = 2; % car on remarque l'existence de deux grands pics dans la variance expliqué

%Calcul des coefficients de la PCA
[coeffs,Xpca,~,~,explained,mu] = pca(image_flat,'NumComponents',nbComponents);

figure;
bar(explained(1:10));
title('Variance expliquée par les composantes principales');
xlabel('Composantes principales');
ylabel('% de variance expliquée');

%Prendre les k premières composantes de la PCA
image_pca = unmask(Xpca,dimx,dimy,nbComponents,mask);

figure;
for i=1:nbComponents
    subplot(2,2,i);
    imagesc(image_pca(:,:,i));
    title('image correspond à chaque composant')
    axis equal;
    colorbar
end


%X = Xpca; % la dimension sera alors 3188*2 
% c_à_d on retient que nbComponents = 4 fonctionnalités.
% donc nos images passe de la dim 128*128*512 à 128*128*4

%% Estimation

[A0,T2] = estimnoise2(image_cropped,params.EchoTime);
A0 = A0';
T2 = T2';
figure;
subplot(121);
imagesc(unmask(A0(mask),dimx,dimy,1,mask));
title('A0');
colorbar;
axis equal;
subplot(122);
% T2(T2<0) = 0;
% T2(abs(T2)>1.5*10e2) = 0;
% T2(A0 < 200) = 0;
imagesc(unmask(T2(mask),dimx,dimy,1,mask),[0 1.5*10e2]);
title('T2');
colorbar;
axis equal;

Xest = [A0(mask) T2(mask)];


X = [Xpca Xest];
%X = Xpca;
%X = Xest;

%% Normalisation

normalized = false;
%Xm = mean(X,1);
%X = normalize(X,1); % ça ne marche pas pour les versions MAtlab au dessous
%de 2018
%X = X + Xm;
%Xnorm = (X - min(X(:))) ./ (max(X(:)) - min(X(:)));
%normalized = true;

%% Classification

nbofclass = 5;

%% Kmeans classification
figure;
tic
indices = kmeans(X,nbofclass,'Replicates',60); % distance euclidienne par défaut
toc
h = subplot(311);
imagesc(unmask(indices,dimx,dimy,1,mask));
title('K-means Distance euclidienne');
colormap(h, lines(nbofclass));
colorbar;
axis equal;
tic
indices = kmeans(X,nbofclass,'Replicates',60,'Distance','cityblock');
toc
h = subplot(312);
imagesc(unmask(indices,dimx,dimy,1,mask));
title('K-means Distance de Manhattan');
colormap(h, lines(nbofclass));
colorbar;
axis equal;
tic
indices = kmeans(X,nbofclass,'Replicates',60,'Distance','cosine');
toc
h = subplot(313);
imagesc(unmask(indices,dimx,dimy,1,mask));
title('K-means Distance cosinusoïdale');
colormap(h, lines(nbofclass));
colorbar;
axis equal;

%% GMM classification

lambda = 0.001;
maxiter = 1000;

if normalized % les courbes de A0,T2 n'ont de sens que si X est non normalisé
    maxplot = 2;
else
    maxplot = 3;
end

map = colormap(parula);
color_idx = linspace(1,size(map,1),nbofclass+1);
estims = zeros(2,nbofclass);

 figure;

mu_class = []; % selon le nombre de classe
for i=1:nbofclass
    mu_class = [mu_class ; mu]; 
end
for g = 1:4
    %disp('Now computing case ' + string(g));
    switch(g)
        case 1
            tic
            GMModel = fitgmdist(X,nbofclass,'Replicates',30,'Options',statset('MaxIter',maxiter));
            toc
        case 2 
            tic
            GMModel = fitgmdist(X,nbofclass,'Replicates',30,'CovarianceType','diagonal','Options',statset('MaxIter',maxiter));
            toc
        case 3 
            tic
            GMModel = fitgmdist(X,nbofclass,'Replicates',30,'RegularizationValue',lambda,'Options',statset('MaxIter',maxiter));
            toc
        case 4
            tic
            GMModel = fitgmdist(X,nbofclass,'Replicates',30,'CovarianceType','diagonal','RegularizationValue',lambda,'Options',statset('MaxIter',maxiter));
            toc
    end
    
    indices = cluster(GMModel,X);
    %if g == 3
        %indices_gmm = indices;
    %end
    
    %subplot(1,maxplot,1);
    h = subplot(2,2,g)

    im = imagesc(unmask(indices,dimx,dimy,1,mask));
    switch(g)
        case 1
            title('GMM Covariance anisotrope');
        case 2
            title('GMM Covariance isotrope (diagonale)');
        case 3
            title('Covariance anisotrope regularisée');
        case 4
            title('Covariance isotrope (diagonale) régularisée');
    end
    colormap(h, lines(nbofclass));
    colorbar;
    axis equal;
    axis equal;
    colormap parula;
end
%% Estimation des temps de relaxation des classes (classification par GMM)

figure;
% Traçage de l'image classifiée :
GMModel = fitgmdist(X,nbofclass,'Replicates',30,'CovarianceType','diagonal','Options',statset('MaxIter',maxiter));
indices = cluster(GMModel,X);
subplot(1,3,1);
im = imagesc(unmask(indices,dimx,dimy,1,mask));
title('GMM Covariance diagonale');
centroids = zeros(nbofclass,2);
% Traçage des distributions des classes :
subplot(1,3,2)
hold on;
for i=1:nbofclass
        plot(X(indices==i,1),X(indices==i,2),'.','Color',map(floor(color_idx(i+1)),:));
        centroids(i,1) = mean(X(indices==i,1));
        centroids(i,2) = mean(X(indices==i,2));
        plot(centroids(i,1),centroids(i,2),'*r');
        xlabel('Première composante principale');
        ylabel('Deuxième composante principale'); 
end
hold off;
% Traçage des courbes de décroissance caractéristique de chaque classe

subplot(1,3,3)
hold on;
x = centroids*coeffs'+mu_class;
for i=1:nbofclass
    plot(x(i,:),'Color',map(floor(color_idx(i)),:));
    %[~,m,b] = regression(1:200,log(x(i,1:200)));
    %estims(:,i) = [-1/m exp(b)];
    %xreg = exp(m*(1:512)+b);
    %plot(xreg,'--','Color',map(floor(color_idx(i)),:));
    title('Courbe de décroissance');
    xlabel('temps');
    ylabel('amplitude');
end
hold off;

%% Markov random fields classification

I = double(image_pca);

figure;

potential     = 0;
maxIter       = 1000;
seg=ICM(I,nbofclass,potential,maxIter);
[width,height,bands]=size(I);
seg=reshape(seg,[width height]);
h = subplot(2,2,1);
imagesc(seg);
title('MRF,paramètre de corrélation = 0 ');
colormap(h, lines(nbofclass));
colorbar;
axis equal;

potential     = 0.6;
seg = ICM(I,nbofclass,potential,maxIter);
[width,height,bands]=size(I);
seg=reshape(seg,[width height]);
h = subplot(2,2,2);
imagesc(seg);
title('MRF,paramètre de corrélation = 0.6 ');
colormap(h, lines(nbofclass));
colorbar;
axis equal;

potential     = 1;
seg=ICM(I,nbofclass,potential,maxIter);
[width,height,bands]=size(I);
seg=reshape(seg,[width height]);
h = subplot(2,2,3);
imagesc(seg);
title('MRF,paramètre de corrélation = 1 ');
colormap(h, lines(nbofclass));
colorbar;
axis equal;

potential     = 4;
seg=ICM(I,nbofclass,potential,maxIter);
[width,height,bands]=size(I);
seg=reshape(seg,[width height]);
h = subplot(2,2,4);
imagesc(seg);
title('MRF,paramètre de corrélation = 4 ');
colormap(h, lines(nbofclass));
colorbar;
axis equal;

%% Estimation des temps de relaxation des classes (classification par MRF)

figure;
% Traçage de l'image classifiée :

GMModel = fitgmdist(X,nbofclass,'Replicates',30,'CovarianceType','diagonal','Options',statset('MaxIter',maxiter));
indices = cluster(GMModel,X);

subplot(1,3,1);
potential     = 0.6;
seg = ICM(I,nbofclass,potential,maxIter);
seg_img_MRF = reshape(seg,[width height]);
imagesc(seg_img_MRF);
title('MRF,paramètre de corrélation = 0.6');

I_MRF=imstack2vectors(I);
[mu,sigma]=GMM_parameter(I_MRF,seg,nbofclass);

% Traçage des distributions des classes :

subplot(1,3,2)
hold on;
for i=1:nbofclass
        plot(X(indices==i,1),X(indices==i,2),'.','Color',map(floor(color_idx(i+1)),:));
        plot(mu(i,1),mu(i,2),'*r');
        xlabel('Première composante principale');
        ylabel('Deuxième composante principale'); 
end
hold off;

% Traçage des courbes de décroissance caractéristique de chaque classe

if ~normalized
    subplot(1,3,3)
    hold on;
    x = mu*coeffs'+mu_class;
    for i=1:nbofclass
        plot(x(i,:),'Color',map(floor(color_idx(i)),:));
        title('Courbe de décroissance MRF');
        xlabel('temps');
        ylabel('amplitude');
    end
    hold off;
end

%% Utilisation de la NMF au lieu de l'ACP :

%Calcul des coefficients de la NMF

[X_nmf,coeffs_nmf] = nnmf(image_flat,nbComponents);

image_nmf = unmask(X_nmf,dimx,dimy,nbComponents,mask);

figure;
for i=1:nbComponents
    subplot(2,2,i);
    imagesc(image_nmf(:,:,i));
    title('image correspond à chaque composant')
    axis equal;
    colorbar
end

% Création de notre espace de descripteur : 

X_NMF = [X_nmf Xest];

I_nmf = double(image_nmf);

figure;

% Traçage de l'image classifiée :

GMModel = fitgmdist(X_NMF,nbofclass,'Replicates',30,'CovarianceType','diagonal','Options',statset('MaxIter',maxiter));
indices = cluster(GMModel,X_NMF);

subplot(1,3,1);
potential     = 0.6;
seg = ICM(I_nmf,nbofclass,potential,maxIter);
seg_img_NMF = reshape(seg,[width height]);
imagesc(seg_img_NMF);
title('MRF-NMF,paramètre de corrélation = 0.6');

I_MRF = imstack2vectors(I_nmf);
[mu_NMF,sigma_NMF]=GMM_parameter(I_MRF,seg,nbofclass);

% Traçage des distributions des classes :

subplot(1,3,2)
hold on;
for i=1:nbofclass
        plot(X_NMF(indices==i,1),X_NMF(indices==i,2),'.','Color',map(floor(color_idx(i+1)),:));
        plot(mu_NMF(i,1),mu_NMF(i,2),'*r');
        xlabel('Première composante principale');
        ylabel('Deuxième composante principale'); 
end
hold off;

% Traçage des courbes de décroissance caractéristique de chaque classe

if ~normalized
    subplot(1,3,3)
    hold on;
    x = mu_NMF*coeffs_nmf+mu_class;
    for i=1:nbofclass
        plot(x(i,:),'Color',map(floor(color_idx(i)),:));
        title('Courbe de décroissance MRF-NMF');
        xlabel('temps');
        ylabel('amplitude');
    end
    hold off;
end

%% Utilisation de ICA au lieu de ACP : 

%Calcul des coefficients de la ICA

[coeffs_ica,X_ica1,X_ica2,mu_ica] = fastICA(image_flat,nbComponents);
coeffs_ica = coeffs_ica';
X_ica =  X_ica2 \ X_ica1';

image_ica = unmask(X_ica,dimx,dimy,nbComponents,mask);

figure;
for i=1:nbComponents
    subplot(2,2,i);
    imagesc(image_ica(:,:,i));
    title('image correspond à chaque composant ICA')
    axis equal;
    colorbar
end

% Création de notre espace de descripteur : 

X = [X_ica Xest];

I_ica = double(image_ica);

figure;

% Traçage de l'image classifiée :

GMModel = fitgmdist(X,nbofclass,'Replicates',30,'CovarianceType','diagonal','Options',statset('MaxIter',maxiter));
indices = cluster(GMModel,X);

subplot(1,3,1);
potential     = 0.6;
seg = ICM(I_ica,nbofclass,potential,maxIter);
seg_img_ICA = reshape(seg,[width height]);
imagesc(seg_img_ICA);
title('MRF-ICA,paramètre de corrélation = 0.6');

I_MRF = imstack2vectors(I_ica);
[mu,sigma]=GMM_parameter(I_MRF,seg,nbofclass);

% Traçage des distributions des classes :

subplot(1,3,2)
hold on;
for i=1:nbofclass
        plot(X(indices==i,1),X(indices==i,2),'.','Color',map(floor(color_idx(i+1)),:));
        plot(mu(i,1),mu(i,2),'*r');
        xlabel('Première composante principale ICA');
        ylabel('Deuxième composante principale ICA'); 
end
hold off;

% Traçage des courbes de décroissance caractéristique de chaque classe

if ~normalized
    subplot(1,3,3)
    hold on;
    x_ica = mu*coeffs_ica' + mu_class;
    for i=1:nbofclass
        plot(x_ica(i,:),'Color',map(floor(color_idx(i)),:));
        title('Courbe de décroissance MRF-ICA');
        xlabel('temps');
        ylabel('amplitude');
    end
    hold off;
end

%% Utilisation de la PCA + ICA : 

%Calcul des coefficients de la ICA

[coeffs_pca_ica,X_ica1,X_ica2,mu_ica] = fastICA(Xpca,nbComponents);
coeffs_pca_ica = coeffs_pca_ica';
X_pca_ica =  X_ica2 \ X_ica1';

image_pca_ica = unmask(X_pca_ica,dimx,dimy,nbComponents,mask);

figure;
for i=1:nbComponents
    subplot(2,2,i);
    imagesc(image_pca_ica(:,:,i));
    title('image correspond à chaque composant PCA + ICA')
    axis equal;
    colorbar
end

% Création de notre espace de descripteur : 

X = [X_pca_ica Xest];

I_pca_ica = double(image_pca_ica);

figure;

% Traçage de l'image classifiée :

GMModel = fitgmdist(X,nbofclass,'Replicates',30,'CovarianceType','diagonal','Options',statset('MaxIter',maxiter));
indices = cluster(GMModel,X);

potential     = 0.6;
seg = ICM(I_pca_ica,nbofclass,potential,maxIter);
seg_img_pca_ICA = reshape(seg,[width height]);
imagesc(seg_img_pca_ICA);
title('MRF-PCA+ICA,paramètre de corrélation = 0.6');

%% Affichage des différentes méthodes : 

figure;
subplot(2,3,1)
imagesc(unmask(indices,dimx,dimy,1,mask));
title('K-means Distance euclidienne');

subplot(2,3,2)
im = imagesc(unmask(indices,dimx,dimy,1,mask));
title('GMM Covariance diagonale');

subplot(2,3,3)
imagesc(seg_img_MRF);
title('MRF,paramètre de corrélation = 0.6');

subplot(2,3,4)
imagesc(seg_img_NMF);
title('MRF-NMF,paramètre de corrélation = 0.6');

subplot(2,3,5)
imagesc(seg_img_ICA);
title('MRF-ICA,paramètre de corrélation = 0.6');

subplot(2,3,6)
imagesc(seg_img_pca_ICA);
title('MRF-PCA+ICA,paramètre de corrélation = 0.6');
