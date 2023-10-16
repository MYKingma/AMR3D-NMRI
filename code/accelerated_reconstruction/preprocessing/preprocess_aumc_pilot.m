%{
    Script for reconstructing pilot scans

    Generates the following:
    - Estimated sensemap
    - K-space
    - Im-space
    - Target (zero-filled reconstruction)
    - PICS reconstruction
    - Single shot data extraction, including
        - Shot pattern
        - Single shot K-space
        - Zero-filled reconstructions
        - PICS reconstuctions


    Script should be runned for all scans seperately
    All results will be saved within subdirectories in procdir

    Author: M.Y. Kingma
    Aknowledgements: M.W.A. Caan, D. Karkalousos
%}

%% Clear all variables and close all figures

clear all;
close all;
clc;

%% Configure paths

filename = '/filename';
sensemapFilename = '/sensemapFilename';

setenv('TOOLBOX_PATH', 'path/to/bart/toolbox')
addpath(getenv('path/to/bart/toolbox'));

maindir = 'path/to/data/dir';
procdir = 'path/to/output/dir';
datedir = '/pilotdir';
subdir = '/sessiondir';
data_dir = append(maindir, datedir, subdir);
raw_data_path = strcat(append(maindir, datedir, subdir), append(filename, '.lab'));


%% Run MRecon 

r = MRecon(raw_data_path);
r.Parameter.Parameter2Read.typ = 1;
r.Parameter.Parameter2Read.Update;
r.ReadData;
r.Parameter.Encoding.XRes=r.Parameter.Encoding.XReconRes;
    
r.RandomPhaseCorrection;
r.RemoveOversampling;
r.PDACorrection;
r.DcOffsetCorrection;
r.MeasPhaseCorrection;
r.SortData;
% r.GridderCalculateTrajectory;
% r.GridData;
r.RingingFilter;
  
r.ZeroFill;   
    
%% Run MRsense
% calculated sensemaps will be stored in: s.Sensitivities
sense_path = strcat(data_dir, append(sensemapFilename, '.lab')); % .lab

s = MRsense(sense_path, r);
sz=size(r.Data);
sz=sz([1 2 8 4]);
s.OutputSizeReformated=sz; %size(r.Data);
s.OutputSizeSensitivity=s.OutputSizeReformated;

s.Mask = 1;
s.Smooth = 1;
s.Extrapolate = 1;
s.Perform;
r.Parameter.Recon.Sensitivities = s;
r.Parameter.Recon.SENSERegStrength = 0;

%% Create sensemap

% t2
smap = bart('caldir 30',bart('fft 7', permute(squeeze(s.ReformatedCoilData), [1 2 4 3]))); 


%% Save kspace, imspace and target

% t2
kspace = permute(squeeze(r.Data), [1 2 4 3]);
imspace = bart('fftshift 2', bart('fft -i 3', kspace));
target = sum(imspace.*conj(smap),4);

dipshow(target)

if contains(filename, 'sag')
    smap_filename = 'smap_sag';
else
    smap_filename = 'smap';
end

writecfl(fullfile(append(procdir, '/kspace', datedir), filename), kspace);
writecfl(fullfile(append(procdir, '/smap', datedir), smap_filename), smap);
writecfl(fullfile(append(procdir, '/target', datedir), filename), target);

%% PICS reconstruction
kspace_pics = permute(squeeze(r.Data), [4 1 2 3]); % slices, x, y, coils

smap_pics = permute(smap, [3 1 2 4]); % slices, x, y, coils
smap_pics = fftshift(smap_pics,3);

pics_recon = [];
for i=1:size(kspace_pics, 1)
    tmp = bart('pics -S -R W:7:0:0.05 -i 20', kspace_pics(i, :, :, :), smap_pics(i, :, :, :));
    
    tmp = permute(fftshift(tmp,3), [1 3 2]);
    pics_recon = [pics_recon; tmp];
end
pics_recon = permute(pics_recon, [3 2 1]);

writecfl(fullfile(append(procdir, '/pics', datedir), filename), pics_recon);

%% Single shot data preparation

vol=squeeze(r.Data);
sz=size(vol);
nchan=sz(3);
idx = r.Parameter.Labels.Index.typ == 1;
kylab=r.Parameter.Labels.Index.ky_label(idx);
ky=r.Parameter.Labels.Index.ky(idx);r.Parameter.Parameter2Read.Update;


% Calculate turbo factor
loca=r.Parameter.Labels.Index.loca;
first_index = find(loca==0,1); %find first zero
diff_loca = diff(loca); %calculate difference between consecutive elements
index = find(diff_loca(first_index:end)~=0,1); %find next non-zero value
result = loca(first_index:first_index+index-2); %get the first zeros
lines_size = size(result(nchan:end));
turbo_factor = lines_size(1);
loca_slice = loca(nchan + 1:end);

% FIXME: assuming to take ceiling operator to center data in k-space!!!
kylab_c=ky(1:nchan:end)+ceil(sz(2)/2)+1;
kylab_c_size = size(kylab_c);
max_idx = kylab_c_size(1) ./ turbo_factor;

% FIXME: this needs to loop over sequentially scanned slices in multislice imaging
tmp = zeros(turbo_factor, 1);
vol_shot_idx = 1;
num_slices = sz(4);
max_shot_idx = max_idx / num_slices;
slice = 1;
max_train_slice = 0;

%% Get shot patterns

% Declare volume for slice
vol_shots_slice = zeros(sz(1), sz(2), sz(3), max_shot_idx); % slice, x, y, shot_idx

% Save indices for later
shot_indices = repmat(zeros(1, turbo_factor), max_shot_idx, 1);

% Loop over shots to obtain shot pattern
for shot_idx = 0:max_idx - 1

    % Declare low and high index in data for y location
    low = (turbo_factor*shot_idx)+1;
    high = low+turbo_factor-1;

    % Get indices of y locations for shot
    idx_shot=kylab_c(low:high);

    % Process data if new y lcoations
    if tmp ~= idx_shot

        % Change y locations
        tmp = idx_shot;
        
        shot_indices(vol_shot_idx, :) = idx_shot;

        % Init shot volume for data (x, y, coils)
        vol_shot=zeros(sz(1:3));

        % Add single shot data for corresponding y locations
        vol_shot(:,idx_shot,:)=vol(:,idx_shot,:,slice);

        % Add to volume for slice
        vol_shots_slice(:,:,:,vol_shot_idx)=vol_shot;

        % End loop if all shots are processed for slice
        if max_shot_idx == vol_shot_idx
            break;
        end

        % Increase shot index
        vol_shot_idx = vol_shot_idx + 1;
    end
end

% Binarize the shots to create a mask
masks_shots = zeros(sz(1), sz(2), max_shot_idx);
for shot_idx = 1:max_shot_idx
    masks_shots(:, :, shot_idx) = vol_shots_slice(:, :, 1, shot_idx) ~= 0;
end

writecfl(fullfile(append(procdir, '/masks', datedir), filename), masks_shots);

%% Get single shot volumes from full k-space

% Create volume of single shot data, per slice
vol_shots_all = zeros(num_slices, sz(1), sz(2), sz(3), max_shot_idx); % slice, x, y, coils, shot_idx

% Loop over kspace slices
for slice = 1:num_slices
    
    slice_data = squeeze(kspace(:, :, slice, :));

    % Loop over kspace shots
    for shot_idx = 1:max_shot_idx
        
        % Get y locations shot
        indices_shot = shot_indices(shot_idx, :);
        
        % Define empty slice
        shot_data = zeros(sz(1), sz(2), sz(3));
        
        % Extract kspace shot data for slice
        shot_data(:, indices_shot, :) = slice_data(:, indices_shot, :);

        % Add kspace shot data to new shot volume
        vol_shots_all(slice, :, :, :, shot_idx) = shot_data;

    end

end

writecfl(fullfile(append(procdir, '/shot_kspace', datedir), filename), vol_shots_all);

permuted_vol_shots = permute(vol_shots_all, [2, 3, 1, 4, 5]);
for shot_idx = 1:max_shot_idx
    shot_filename = strcat(filename,  '_shot_', num2str(shot_idx));
    writecfl(fullfile(append(procdir, '/shot_kspace', datedir), shot_filename), permuted_vol_shots(:, :, :, :, 1));
end
    

%% Reconstruct single shot volumes - PICS

vol_shot_recon_pics = zeros(num_slices, sz(1), sz(2), max_shot_idx); % slice, x, y, shot_idx

for slice = 1:num_slices
    
    for shot_idx = 1:max_shot_idx
        pics_recon_shot = bart('pics -S -R W:7:0:0.05 -i 20', vol_shots_all(slice, :, :, :, shot_idx), smap_pics(slice, :, :, :));
        
        pics_recon_shot = permute(fftshift(pics_recon_shot,3), [1 3 2]);
        vol_shot_recon_pics(slice, :, :, shot_idx) = squeeze(permute(pics_recon_shot, [3 2 1]));
    end
end

shot_to_plot = 1;

vol_shot_recon_pics = permute(vol_shot_recon_pics(:, :, :, shot_to_plot), [2 3 1 4]);

writecfl(fullfile(append(procdir, '/shot_pics', datedir), filename), vol_shot_recon_pics);

%% Reconstruct single shot volumes - Default

vol_shot_recon_imspace = zeros(sz(1), sz(2), num_slices, sz(3), max_shot_idx); % slice, x, y, shot_idx
vol_shot_recon_target = zeros(sz(1), sz(2), num_slices,  max_shot_idx); % slice, x, y, shot_idx


for shot_idx = 1:max_shot_idx
   imspace_shot =  bart('fftshift 2', bart('fft -i 3', permute(vol_shots_all(:, :, :, :, shot_idx), [2 3 1 4])));
   target_shot = sum(imspace_shot.*conj(smap),4);
   vol_shot_recon_target(:, :, :, shot_idx) = target_shot;
   vol_shot_recon_imspace(:, :, :, :, shot_idx) = imspace_shot;
end

shot_to_plot = 1;

writecfl(fullfile(append(procdir, '/shot_target', datedir), filename), vol_shot_recon_target);
