%{
    Script which manually upsamples neonatal images to isotropic resolution
    Using the SPM toolbox to coregister anisotropic images to an isotropic image

    Author: M.W.A Caan
%}

% Define the path to the data directory
data_path = '/path/to/data';
gt_path = '/path/to/data';
output_path = '/path/to/data';
gt_output_path = '/path/to/data';

% Add module paths
addpath('path/to/matlab/')
addpath('path/to/spm/toolbox')
addpath('path/to/dipimage/toolbox')

% Get list of directories in recon_path
dir_list = dir(data_path);
dir_list = dir_list([dir_list.isdir]);
dir_list = dir_list(~ismember({dir_list.name},{'.','..'}));

% Loop over each directory
for i = 1:length(dir_list)
    % Get the filepath of the downsampled file inside each directory containing `downsampled_0` in the filename
    downsampled_path = fullfile(data_path, dir_list(i).name, strcat('*downsampled_0*.nii*'));
    downsampled_file = dir(downsampled_path);
    downsampled_file_path = fullfile(downsampled_file.folder, downsampled_file.name);

    % Get the filepath of the output file inside each directory containing `output` in the filename, if not found, skip this directory
    recon_output_path = fullfile(data_path, dir_list(i).name, strcat('*output*.nii.gz'));
    recon_file = dir(recon_output_path);
    if isempty(recon_file)
        continue
    end
    recon_file_path = fullfile(recon_file.folder, recon_file.name);

    % Get the filepath of the gt, this is inside the gt_path. The filename inside the gt_path is the same as the dirname inside the data_path (suffixed with nii.gz)
    gt_file_name = strcat(dir_list(i).name, '.nii.gz');
    gt_file_path = fullfile(gt_path, gt_file_name);

    % Unzip the gt and output file
    gunzip(gt_file_path);
    gunzip(recon_file_path);
    
    % Get the path to the unzipped gt and output file
    unzipped_gt_file_path = fullfile(gt_path, [dir_list(i).name '.nii']);
    unzipped_output_file_path = fullfile(data_path, dir_list(i).name, strcat(dir_list(i).name, '_output.nii'));
    
    % Run the function `coregister_to_epi()
    coregister_to_epi(unzipped_gt_file_path, unzipped_output_file_path);
    coregister_to_epi(downsampled_file_path, unzipped_output_file_path);

    % Coregister_to_epi creates new files inside the data dir, prefixed with r, get this file, and move it to the output_path and gt_output_path
    coreg_gt_file_name = strcat('r', dir_list(i).name, '.nii');
    coreg_down_file_name = strcat('r', downsampled_file.name);
    
    % Get new path to the coregistered files
    coregistered_gt_file_path = fullfile(gt_path, coreg_gt_file_name);
    coregistered_down_file_path = fullfile(data_path, dir_list(i).name, coreg_down_file_name);

    % Move the coregistered files to the output_path
    movefile(coregistered_gt_file_path, gt_output_path);
    movefile(coregistered_down_file_path, output_path);
    
    % Delete the prefix r from the filename
    new_gt_file_name = strrep(coreg_gt_file_name, 'r', '');
    new_down_file_name = strrep(coreg_down_file_name, 'r', '');

    % Rename the files
    movefile(fullfile(gt_output_path, coreg_gt_file_name), fullfile(gt_output_path, new_gt_file_name));
    movefile(fullfile(output_path, coreg_down_file_name), fullfile(output_path, new_down_file_name));
    
    % Delete the unzipped files
    delete(unzipped_gt_file_path);
    delete(unzipped_output_file_path);

end
