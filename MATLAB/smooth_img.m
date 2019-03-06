function img = smooth_img(img,fwhm,vx)
% Gaussian smoothing (in memory), taking into account voxel anisotropy.
%_______________________________________________________________________
%  Copyright (C) 2018 Wellcome Trust Centre for Neuroimaging

if nargin < 2, fwhm = 1; end
if nargin < 3, vx   = 1; end

if numel(fwhm) == 1, fwhm = fwhm*ones(1,3); end
if numel(vx) == 1,   vx = vx*ones(1,3); end

if fwhm > 0        
    fwhm = fwhm./vx;            % voxel anisotropy
    s1   = fwhm/sqrt(8*log(2)); % FWHM -> Gaussian parameter

    x  = round(6*s1(1)); x = -x:x; x = spm_smoothkern(fwhm(1),x,1); x  = x/sum(x);
    y  = round(6*s1(2)); y = -y:y; y = spm_smoothkern(fwhm(2),y,1); y  = y/sum(y);
    z  = round(6*s1(3)); z = -z:z; z = spm_smoothkern(fwhm(3),z,1); z  = z/sum(z);

    i  = (length(x) - 1)/2;
    j  = (length(y) - 1)/2;
    k  = (length(z) - 1)/2;

    spm_conv_vol(img,img,x,y,z,-[i,j,k]);   
end
%==========================================================================