run vlfeat/toolbox/vl_setup

run matconvnet-1.0-beta9/matlab/vl_setupnn
addpath matconvnet-1.0-beta9/examples/
addpath matlab-helpers
clear mex ;

%{
vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc', ...
'cudaRoot', '/usr/local/cuda', ...
'enableCudnn', true, 'cudnnRoot', '/home/arunirc/Downloads/cudnn-7.0-v3', ...
'enableImreadJpeg', true, 'Verbose', 1) ;
run matlab/vl_setupnn
%}