% clear; clc;
% 
% % wi
% DirData = '/run/media/mbrud/iomega/WORK/Modalities/IMG/';
% Nii_wi  = nifti(spm_select('FPListRec',DirData,'^.*\.nii$'));
% S0      = numel(Nii_wi);
% 
% % wc
% DirData = {'/run/media/mbrud/iomega/WORK/Modalities/wGM/', ...
%            '/run/media/mbrud/iomega/WORK/Modalities/wWM/', ...
%            '/run/media/mbrud/iomega/WORK/Modalities/wOTH/'};       
% Nii_wc  = cell(1,3);
% for i=1:numel(DirData)       
%     Nii_wc{i} = nifti(spm_select('FPListRec',DirData{i},'^.*\.nii$'));
% end
% 
% % mwc
% DirData = {'/run/media/mbrud/iomega/WORK/Modalities/mwGM/', ...
%            '/run/media/mbrud/iomega/WORK/Modalities/mwWM/', ...
%            '/run/media/mbrud/iomega/WORK/Modalities/mwOTH/'};       
% Nii_mwc = cell(1,3);
% for i=1:numel(DirData)       
%     Nii_mwc{i} = nifti(spm_select('FPListRec',DirData{i},'^.*\.nii$'));
% end

%% Parameters
S      = S0;
Samp   = 1;
Degree = 4;
FWHM   = 0;

%% wi
DirDS = '../Data/DownSampled-wi-full';
if exist(DirDS,'dir') == 7, rmdir(DirDS,'s'); end; mkdir(DirDS);

for s=1:S
    fprintf('%i ',s)
    
    [img,mat,dm] = resample_img(Nii_wi(s),Samp,Degree);    
    
    [~,nam,ext] = fileparts(Nii_wi(s).dat.fname);
    FileName    = fullfile(DirDS,[nam ext]);
    
    create_nii(FileName,img,mat,Nii_wi(s).dat.dtype,'Down-sampled');
end
fprintf('\nDone!\n')

%% wc
DirDS = '../Data/DownSampled-wc-full';
if exist(DirDS,'dir') == 7, rmdir(DirDS,'s'); end; mkdir(DirDS);

for s=1:S
    fprintf('%i ',s)
    
    simg = 0;
    img  = cell(1,numel(Nii_wc));
    for i=1:numel(Nii_wc)
        [img{i},mat,dm]  = resample_img(Nii_wc{i}(s),Samp,Degree);
        img{i}(img{i}<0) = 0;
        simg             = simg + img{i};
    end
    
    pth  = fileparts(Nii_wc{1}(s).dat.fname);
    pth  = strsplit(pth,filesep);
    Name = pth{end};
    
    Dir = fullfile(DirDS,Name);
    mkdir(Dir)
    
    for i=1:numel(Nii_wc)
        img{i} = img{i}./(simg + eps);
        vx     = sqrt(sum(mat(1:3,1:3).^2));
        img{i} = smooth_img(img{i},FWHM,vx);
    
        [~,nam,ext] = fileparts(Nii_wc{i}(s).dat.fname);
        FileName    = fullfile(Dir,[nam ext]);

        create_nii(FileName,img{i},mat,Nii_wc{i}(s).dat.dtype,'Down-sampled');
    end   
end
fprintf('\nDone!\n')

%% wc
DirDS = '../Data/DownSampled-mwc-full';
if exist(DirDS,'dir') == 7, rmdir(DirDS,'s'); end; mkdir(DirDS);

for s=1:S
    fprintf('%i ',s)
    
    pth  = fileparts(Nii_mwc{1}(s).dat.fname);
    pth  = strsplit(pth,filesep);
    Name = pth{end};
    
    Dir = fullfile(DirDS,Name);
    mkdir(Dir)
    
    for i=1:numel(Nii_mwc)
        [img,mat,dm]  = resample_img(Nii_mwc{i}(s),Samp,Degree);
        vx            = sqrt(sum(mat(1:3,1:3).^2));
        img           = smooth_img(img,FWHM,vx);
        
        [~,nam,ext] = fileparts(Nii_mwc{i}(s).dat.fname);
        FileName    = fullfile(Dir,[nam ext]);

        create_nii(FileName,img,mat,Nii_mwc{i}(s).dat.dtype,'Down-sampled');
    end
end
fprintf('\nDone!\n')

%%
if 0
    files = spm_select('FPListRec','/home/mbrud/dev/Validations/CNN-regress/Data/DownSampled-wc-full','^wc2.*\.nii$');
    
    DirDS = '../Data/wgm';
    if exist(DirDS,'dir') == 7, rmdir(DirDS,'s'); end; mkdir(DirDS);
    
    for s=1:size(files,1)
        FileName = strtrim(files(s,:));
        movefile(FileName,DirDS)
    end
end