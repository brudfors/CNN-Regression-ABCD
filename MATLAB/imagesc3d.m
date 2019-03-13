function imagesc3d(img,varargin)

Spacing = 8;        % Step size
dm      = size(img);
dm      = [dm 1];

if dm(3) > 1
    % Montage parameters
    z  = Spacing:Spacing:dm(3);
    z  = z(1:end - 2);
    N  = numel(z);    
    nr = floor(sqrt(N));
    nc = ceil(N/nr);  
    
    % Create montage
    mtg = zeros([nr*dm(1) nc*dm(2)],'single');
    cnt = 1;
    for r=1:nr
        for c=1:nc
            if cnt > numel(z)
                break
            end
            
            mtg(1 + (r - 1)*dm(1):r*dm(1),1 + (c - 1)*dm(2):c*dm(2)) = img(:,:,z(cnt));
            
            cnt = cnt + 1;
        end
    end   
    img = mtg;
end

imagesc(img,varargin{:});
%==========================================================================