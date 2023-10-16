%{
    Script which preprocesses the Utrecht piot data, used for generating sensemaps from .cpx files.

    Author: D. Karkalousos and M.W.A Caan
%}

%%%%%%%% Preprocess Utrecht pilot data, generate sagittal sensemap, author: D. Karkalousos
%% Clear all variables and close all figures

clear all;
close all;
clc;

%% Configure paths

setenv('TOOLBOX_PATH', 'path/to/bart/toolbox')
addpath(getenv('path/to/bart/toolbox'));
addpath(('path/to/spm/toolbox'));

parentraw='/path/to/raw/data/';
parentproc='/path/to/proc/data/';
subdirs={'subdir_name'};
niifile='/path/to/nii/file.nii';

doPreproc= 1; % should preproc be performed again? time-consuming

%% Prepare files
iDir=1;

rawdir=fullfile(parentraw,subdirs{iDir});
odir=fullfile(parentproc,subdirs{iDir});
if ~isfolder(odir); mkdir(odir); end
disp(['subject ',num2str(iDir),', folder: ',rawdir]);

ref= dir([rawdir,'/*sense.cpx']);
ref=fullfile(rawdir,ref(1).name);
refsin=dir([rawdir,'/*sense.sin']);
refsin=fullfile(rawdir,refsin(1).name);
t2sin=dir([rawdir,'/*scan.sin']);

if length(t2sin)==1
    t2sin=fullfile(rawdir,t2sin(1).name);
else
    t2sin=fullfile(rawdir,t2sin(2).name);
end

t2=dir([rawdir,'/*scan.raw']);
t2=fullfile(t2.folder, t2.name);

rfile=fullfile(odir,'senserefall_real.nii');
ifile=fullfile(odir,'senserefall_imag.nii');
rrfile=fullfile(odir,'rsenserefall_real.nii');
rifile=fullfile(odir,'rsenserefall_imag.nii');

%% Run MRecon 

r = MRecon(t2);
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
r.RingingFilter;

%% Create matrix from sin file
rdata = permute(squeeze(r.Data(:,:,1,:,1,1,1,:)), [1 2 4 3]);

sx=size(rdata,1); sy=size(rdata,2); sz=size(rdata,3); nrcoils=size(rdata,4);

rrz_os=sz;

[~, rrx]=matrix_from_sin(t2sin, rrz_os);
rry=rrx(2); rrz=rrx(3); rrx=rrx(1);

vol3=zeros(rrx, rry, rrz_os, nrcoils); % try padded
vol3(rrx/2 - sx/2 +1:rrx/2 +sx/2 , rry/2 - sy/2 +1:rry/2 +sy/2, round(rrz_os/2 - sz/2) +1:round(rrz_os/2 +sz/2), :) = rdata; 

%% Generate sensemap
disp('loading senseref.cpx and applying transforms');

C=MRecon(ref);
C.ReadData; % read
sensemaps=C.Data;

sensemaps=flip(flip(sensemaps,2),3);
sensemaps=permute(sensemaps,[2 1 3 4 5 6 7 8]);

% Bodycoil
sensemaps_bc=sensemaps(:,:,:,1,1,1,1,2);
sensemaps=squeeze(sensemaps(:,:,:,:,1,1,1,1));

% Remove inactive coils
sensemaps=sensemaps(:,:,:,3:15); % active coil numbers from sin file

sensemaps_size=size(sensemaps);
spatial_dims=sensemaps_size(1:3);

sref_off=[-23.8908      -0.7314      10.0000]; % Y X Z from SIN file: reconstruction
sp=[5.4688       5.4688       4.0000]; % from SIN file: 01 00 00: voxel_sizes

offset = -[spatial_dims(3)*sp(3)/2+sref_off(2) spatial_dims(1)*sp(1)/2+sref_off(1)  spatial_dims(2)*sp(2)/2-sref_off(3)]; % sagittal

% Offcentre: half the number of voxels * voxel spacing; 
a = [offset 0 pi/2 -pi/2 sp(1) sp(2) sp(3) 0 0 0];
A = spm_matrix(a);

%% Prepare files
rfile=fullfile(odir,'senserefall_real.nii');
ifile=fullfile(odir,'senserefall_imag.nii');
rrfile=fullfile(odir,'rsenserefall_real.nii');
rifile=fullfile(odir,'rsenserefall_imag.nii');

%% Save sensemap
n=nifti(niifile);
n.dat.fname=rfile;
n.dat.scl_slope=max(abs(sensemaps(:)))/1e4;
n.mat=A;
n.mat0=n.mat;
n.dat.dim=size(sensemaps);
create(n);
n.dat(:,:,:,:)=real(sensemaps);

n.dat.fname=ifile;
create(n);
n.dat(:,:,:,:)=imag(sensemaps);

%% Reslice to T2-sag
flags.mean=0;
flags.which=1;
spm_reslice({niifile,rfile,ifile},flags)

%% Bring sensemap back to raw data convention
n=nifti(rrfile);
rout=n.dat(:,:,:,:);
n=nifti(rifile);
iout=n.dat(:,:,:,:);

%% It follows from below that the senseref data need to be permuted here
s_resliced=rout+1i*iout; % make complex-valued output

%%
sensemaps=bart('caldir 30',bart('fft 7', s_resliced));
sensemaps_reoriented=flip(flip(permute(sensemaps, [2 1 3 4]), 1), 2);

%%
imspace = bart('fftshift 4', bart('fftshift 2', bart('fft -i 7', vol3)));
target = sum(imspace.*conj(sensemaps_reoriented), 4);


%% Helper functions
function [matrix_perm,resolution]=matrix_from_sin(t2sin, rrz_os)

matrix_loca=zeros(3,3);

[~,offcentr1]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_offcentr_incrs'' | awk ''{print$6}'' ']);
[~,offcentr2]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_offcentr_incrs'' | awk ''{print$7}'' ']);
[~,offcentr3]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_offcentr_incrs'' | awk ''{print$8}'' ']);
matrix_loca(3,:)= [str2double(strip(offcentr1)) str2double(strip(offcentr2)) str2double(strip(offcentr3))] ; clear offcentr1 offcentr2 offcentr3

[~,row1]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$6}'' ']);
[~,row2]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$7}'' ']);
[~,row3]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$8}'' ']);
matrix_loca(1,:)= [str2double(strip(row1)) str2double(strip(row2)) str2double(strip(row3))] ; clear row1 row2 row3

[~,col1]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$6}'' ']);
[~,col2]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$7}'' ']);
[~,col3]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$8}'' ']);
matrix_loca(2,:)= [str2double(strip(col1)) str2double(strip(col2)) str2double(strip(col3))] ; clear col1 col2 col3

[~,vox1]=unix(['cat ',t2sin,' | grep ''voxel_sizes'' | awk ''{print$6}'' ']);
[~,vox2]=unix(['cat ',t2sin,' | grep ''voxel_sizes'' | awk ''{print$7}'' ']);
[~,vox3]=unix(['cat ',t2sin,' | grep ''voxel_sizes'' | awk ''{print$8}'' ']);
voxel_sizes= [str2double(strip(vox1)) str2double(strip(vox2)) str2double(strip(vox3))] ; clear vox1 vox2 vox3

[~,res1]=unix(['cat ',t2sin,' | grep ''output_resolutions'' | awk ''{print$6}'' ']);
[~,res2]=unix(['cat ',t2sin,' | grep ''output_resolutions'' | awk ''{print$7}'' ']);
resolution= [str2double(strip(res1)) str2double(strip(res2)) rrz_os] ; clear res1 res2 res3

[~,cc1]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$6}'' ']);
[~,cc2]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$7}'' ']);
[~,cc3]=unix(['cat ',t2sin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$8}'' ']);
centre_coords= [-str2double(strip(cc1)) -str2double(strip(cc2)) str2double(strip(cc3))] ; clear cc1 cc2 cc3

matrix_loca=matrix_loca';

matrix_loca(1,3)=-matrix_loca(1,3);
matrix_loca(2,3)=-matrix_loca(2,3);
matrix_loca(3,1)=-matrix_loca(3,1);
matrix_loca(3,2)=-matrix_loca(3,2);

matrix(:,1)= matrix_loca(:,1)*voxel_sizes(1);
matrix(:,2)= matrix_loca(:,2)*voxel_sizes(2);
matrix(:,3)= matrix_loca(:,3);

offset2 =  -resolution(1)/2*matrix(1,1) + ...
            -resolution(2)/2*matrix(1,2) + ...
            -resolution(3)/2*matrix(1,3) + ...
            +centre_coords(1);
offset2 =   -resolution(1)/2*matrix(2,1) + ...
            -resolution(2)/2*matrix(2,2) + ...
            -resolution(3)/2*matrix(2,3) + ...
            +centre_coords(2);
offset3 =   -resolution(1)/2*matrix(3,1) + ...
            -resolution(2)/2*matrix(3,2) + ...
            -resolution(3)/2*matrix(3,3) + ...
            +centre_coords(3);

matrix(:,4)= [offset2; offset2; offset3];

matrix_perm = zeros(size(matrix));
 matrix_perm(:,1)= matrix(:,3);
 matrix_perm(:,2)= matrix(:,1);
 matrix_perm(:,3)= matrix(:,2);
 matrix_perm(:,4)= matrix(:,4);

 matrix_perm = matrix_perm([2 1 3],:);
 matrix_perm(4,:)= [0 0 0 1];
end


%%%%%%%% Generate sagittal sensemap, author: M.W.A Caan
% TODO: First copy files rsense* from sag folder

cd /path/to/sag
n=nifti('rsenserefall_real.nii');
v=n.dat(:,:,:,:);
f=flip(v,1);
n.dat.fname=('frsenserefall_real.nii');
create(n);
n.dat(:,:,:,:)=f;

n=nifti('rsenserefall_imag.nii');
v=n.dat(:,:,:,:);
f=flip(v,1);
n.dat.fname=('frsenserefall_imag.nii');
create(n);
n.dat(:,:,:,:)=f;

cd /scratch/mwcaan/neonatal/recon/tra

% TODO: Copy sag file to here

spm_reslice({'scan.nii', ... 
'frsenserefall_real.nii','frsenserefall_imag.nii'})
  
n=nifti('rfrsenserefall_real.nii');
sr=n.dat(:,:,:,:);
n=nifti('rfrsenserefall_imag.nii');
si=n.dat(:,:,:,:);
s=sr+1i.*si;

cd /path/to/data

%% Explore sagittal data
% Imspace and senseref aligned after permute and flip
cd /path/to/data
im=readcfl('imspace');
n=nifti('rsenserefall_real.nii');
sr=n.dat(:,:,:,:);
n=nifti('rsenserefall_imag.nii');
si=n.dat(:,:,:,:);
s=sr+1i.*si;
s=permute(s,[2 1 3 4 ]);
f=flip(s,1);

t1=arr(f(:,88,:,:));
t1=abs(t1);
t1=t1/max(t1(:));
t2=arr(im(:,88,:,:));
t2=abs(t2);
t2=t2/max(t2(:));

rgb(t1,t2)

%% Align senseref to off-scanner nifti
cd path/to/sag
n=nifti('rsenserefall_real.nii');
sr=n.dat(:,:,:,:);
n=nifti('rsenserefall_imag.nii');
si=n.dat(:,:,:,:);
s=sr+1i.*si;
f=flip(s,1); % flip AP
f=flip(f,3); % an extra flip along LR axis is needed?!
f=abs(f);
f=sum(f,4);
f=f/max(f(:));
n.dat.fname='frsenserefall_abs_sag.nii'
n.dat.dim=n.dat.dim(1:3);
n.dat.scl_slope=1e-3;
create(n);
n.dat(:,:,:)=f;

%% Transverse - verify on-scanner nifti matches raw imaging data

cd /path/to/data
im=readcfl('imspace');

niifile='/path/to/nii/file.nii';
n=nifti(niifile);
v=n.dat(:,:,:);
v=permute(v,[2 1 3]);
v=flip(v,1);
v=flip(v,2);
dipshow(v)

%% Reslice sag reg scan to axial
cd /path/to/tra
system('cp ../sag/frsenserefall_abs_sag.nii .')
spm_reslice({'path/to/nii/file.nii'})

n=nifti('rfrsenserefall_abs_sag.nii');
sabs=n.dat(:,:,:);
s=permute(sabs,[2 1 3 4]);
f=flip(s,1);

n=nifti('path/to/nii/file.nii')

cd /path/to/data

im=readcfl('imspace');
