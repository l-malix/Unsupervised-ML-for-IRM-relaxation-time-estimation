function [A0,T2]=estimnoise2(Image,EchoTime)

xdim=size(Image,1);
ydim=size(Image,2);
tdim=size(Image,3);

deb = 1; 
fin = 10;
xi        = zeros(1,1,fin-deb+1);
xi(1,1,:) = EchoTime(deb:fin);
un        = ones(xdim,ydim,fin-deb+1);
xi        = repmat(xi,xdim,ydim,1);
yi        = log(Image(:,:,deb:fin));

%%
N = size(xi,3);
Sx = sum(xi,3);
Sx2 = sum(xi.^2,3);
Sy = sum(yi,3);
Sxy = sum(xi.*yi,3);
den = (N*Sx2 - Sx.^2);
a = (N*Sxy - Sx.*Sy)./den;
b = (Sx2.*Sy - Sx.*Sxy)./den;

A0 = exp(b');
T2 = -1./a';

