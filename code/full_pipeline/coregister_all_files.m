%{
    Script for coregistering slice-to-volume reconstructions with the input stacks

    Author: M.Y. Kingma
    Aknowledgements: M.W.A. Caan, D. Karkalousos
%}


% Define the path to the data directory
data_path = '/path/to/data/dir';
addpath('/path/to/matlab')
addpath('/path/to/matlab/spm')
addpath('/path/to/matlab/dipimage')

% Get a list of all directories in the data directory
dir_list = dir(data_path);
dir_list = dir_list([dir_list.isdir]);
dir_list = dir_list(~ismember({dir_list.name},{'.','..'}));

% Loop over each directory
for i = 1:length(dir_list)
    % Get the name of the current directory
    dir_name = dir_list(i).name;
    
    % Define the path to the current directory
    dir_path = fullfile(data_path, dir_name);
    
    % Get a list of all zip files in the current directory
    zip_list = dir(fullfile(dir_path, '*.gz'));
    
    % Loop over each zip file
    for j = 1:length(zip_list)
        % Unzip the current zip file
        zip_path = fullfile(dir_path, zip_list(j).name);
        gunzip(zip_path, dir_path);
        
        % Remove the zip file
        delete(zip_path);
    end
    
    % Get the file list of files containing 'downsampled' in the current directory
    file_list = dir(fullfile(dir_path, '*input*'));

    % Sort the file list inverse alphabetically
    [~, idx] = sort({file_list.name});
    
    % Loop over each pair of files
    for j = 1:2:length(file_list)
        % Get the filenames of the current pair of files
        filename1 = fullfile(dir_path, file_list(j+1).name);
        filename2 = fullfile(dir_path, file_list(j).name);
        
        % Call the coregister_to_epi function
        coregister_to_epi(filename2, filename1);
        
        % Remove the original file
        delete(filename1);
        
        % Rename the new file
        new_filename = fullfile(dir_path, ['r' file_list(j).name]);
        movefile(new_filename, fullfile(dir_path, file_list(j+1).name));
    end
end