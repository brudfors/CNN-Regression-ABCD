clear; clc;

%% Parameters
S           = Inf;
Degree      = 4;
LoadMask    = true;
LoadNii     = true;
DirData     = '/data/mbrud/Validations/segmentation-model/abcd-3d-all/subject-results';
BGClass     = 1;
DirOut_wi0  = '../Data/wi';
DirOut_wc0  = '../Data/wc';
DirOut_mwc0 = '../Data/mwc';
ShowMask    = false;

Samp0 = [1 1/2 1/2];
FWHM0 = [0   0 12];

%% Get data
if LoadNii
    var    = load('Nii_wi.mat');
    Nii_wi = var.Nii_wi;
    
    var    = load('Nii_wc.mat');
    Nii_wc = var.Nii_wc;
    
    var     = load('Nii_mwc.mat');
    Nii_mwc = var.Nii_mwc;    
    
    clear var
else
    % wi
    Nii_wi = nifti(spm_select('FPListRec',DirData,'^ci.*\.nii$')); % obs! should be wi


    % wc    
    Nii_wc = cell(1,3);
    for i=1:3
        Nii_wc{i} = nifti(spm_select('FPListRec',DirData,['^wc' num2str(i) '.*\.nii$']));
    end

    % mwc    
    Nii_mwc = cell(1,3);
    for i=1:3
        Nii_mwc{i} = nifti(spm_select('FPListRec',DirData,['^mwc' num2str(i) '.*\.nii$']));
    end
    
    % save
    save('Nii_wi.mat', 'Nii_wi')
    save('Nii_wc.mat', 'Nii_wc')
    save('Nii_mwc.mat','Nii_mwc')
end

S0 = numel(Nii_wi);
S  = min(S,S0);

%% Get mask
if ~LoadMask
    smsk0 = 0;
    for s=1:S0
        fprintf('.')

        img   = Nii_wi(s).dat(:,:,:);        
        msk   = img > 0;
        smsk0 = smsk0 + msk;
    end
    fprintf('\n')
    fprintf('Done!\n')
    
    save('smsk0.mat','smsk0')
else
    load('smsk0.mat')
end

if ShowMask
    figure(666); subplot(221); imagesc3d(smsk0); drawnow; axis off
    disp(['max(smsk0) = ' num2str(max(smsk0(:)))])
end

% Threshold mask
CutPrct = 0.99;
Mask    = smsk0 > (CutPrct*S0);
Mask    = imfill(Mask,'holes');

% % Erode
% se   = strel('cube',3);
% % se   = strel('sphere',1);
% Mask = imerode(Mask, se);
% imagesc3d(Mask); drawnow; axis off

if ShowMask
    subplot(222); imagesc3d(Mask); drawnow; axis off

    % Show a masked image
    s   = 1;
    img = Nii_wi(s).dat(:,:,:);
    subplot(223); imagesc3d(img); drawnow; axis off
    img = Mask.*img;
    subplot(224); imagesc3d(img); drawnow; axis off
end

for i=1:numel(Samp0)
    
    Samp = Samp0(i);
    FWHM = FWHM0(i);
    
    %% Output directories
    prefix = '';
    if Samp ~= 1 && Samp ~= 0
        prefix = [prefix '-samp' num2str(Samp)];
    end
    if FWHM > 0
        prefix = [prefix '-fwhm' num2str(FWHM)];
    end
    if ~isempty(prefix)
        prefix = ['-' prefix];
    end

    DirOut_wi  = [DirOut_wi0  prefix];
    DirOut_wc  = [DirOut_wc0  prefix];
    DirOut_mwc = [DirOut_mwc0 prefix];

    if exist(DirOut_wi,'dir') == 7,  rmdir(DirOut_wi,'s'); end; mkdir(DirOut_wi);
    if exist(DirOut_wc,'dir') == 7,  rmdir(DirOut_wc,'s'); end; mkdir(DirOut_wc);
    if exist(DirOut_mwc,'dir') == 7, rmdir(DirOut_mwc,'s'); end; mkdir(DirOut_mwc);

    %% wi
    for s=1:S
        fprintf('%i ',s)

        [img,mat,dm] = resample_img(Nii_wi(s),Mask,Samp,Degree);    

        [~,nam,ext] = fileparts(Nii_wi(s).dat.fname);
        FileName    = fullfile(DirOut_wi,[nam ext]);

        create_nii(FileName,img,mat,Nii_wi(s).dat.dtype,'Down-sampled');
    end
    fprintf('\nDone!\n')

    %% wc
    for s=1:S
        fprintf('%i ',s)

        simg = 0;
        img  = cell(1,numel(Nii_wc));
        for i=1:numel(Nii_wc)
            if i == BGClass
                [img{i},mat,dm]  = resample_img(Nii_wc{i}(s),[],Samp,Degree);
            else
                [img{i},mat,dm]  = resample_img(Nii_wc{i}(s),Mask,Samp,Degree);
            end
            img{i}(img{i}<0) = 0;
            simg             = simg + img{i};
        end

        pth  = fileparts(Nii_wc{1}(s).dat.fname);
        pth  = strsplit(pth,filesep);
        Name = pth{end - 1};

        Dir = fullfile(DirOut_wc,Name);
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

    %% mwc
    for s=1:S
        fprintf('%i ',s)

        pth  = fileparts(Nii_mwc{1}(s).dat.fname);
        pth  = strsplit(pth,filesep);
        Name = pth{end - 1};

        Dir = fullfile(DirOut_mwc,Name);
        mkdir(Dir)

        for i=1:numel(Nii_mwc)
            if i == BGClass
                [img,mat,dm]  = resample_img(Nii_mwc{i}(s),[],Samp,Degree);
            else
                [img,mat,dm]  = resample_img(Nii_mwc{i}(s),Mask,Samp,Degree);
            end
            vx            = sqrt(sum(mat(1:3,1:3).^2));
            img           = smooth_img(img,FWHM,vx);

            [~,nam,ext] = fileparts(Nii_mwc{i}(s).dat.fname);
            FileName    = fullfile(Dir,[nam ext]);

            create_nii(FileName,img,mat,Nii_mwc{i}(s).dat.dtype,'Down-sampled');
        end
    end
    fprintf('\nDone!\n')
end