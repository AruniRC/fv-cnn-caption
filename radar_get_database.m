function [imdb, metadata] = radar_get_database(writeScanImage, varargin)
%RADAR_GET_DATABASE Read in NARRstyle radar data and render images.
%   Renders reflectivity for a single sweep as a single-channel JPG image.
%
% INPUT
%   writeScanImage      TRUE/FALSE depending on whether to write rendered
%                       scans to file or load in pre-computed metadata. 
%                       If a rendered images already exists
%                       in that location, then it is NOT over-written.
%
% OPTIONS
%   radarPath           Location of the root folder with radar data as *.gz
%                       files.
%
%   renderPath          Location where to write rendered images, if
%                       writeScanImage flag is set to TRUE.
%
%   sweep               Specify which individual sweep of the scan is to be
%                       rendered.
%
%   radarLibPath        Root folder path where projects rscancv, wsrlib and
%                       rclass2014 from MLDS are saved. These projects are
%                       dependencies that are called to render the scans as
%                       images.

opts.seed = 0;
opts.renderPath = 'data/render-full';
% opts.radarPath = '/home/arunirc/Dropbox/NARRstyle/radar/data';
opts.radarPath = '/scratch3/arunirc/radar_project/radar_data/data';
opts.radarLibPath = '/scratch3/arunirc/radar_project';
opts.sweep = 1;
opts.mode = 'dz';

opts = vl_argparse(opts, varargin);
rng(opts.seed) ;

% add paths to scan rendering code
RADAR_PATH = opts.radarLibPath;
addpath(genpath(fullfile(RADAR_PATH, 'rclass2014')));
addpath(genpath(fullfile(RADAR_PATH, 'rscancv')));
addpath(genpath(fullfile(RADAR_PATH, 'wsrlib')));


if writeScanImage
    fprintf('Rendering scans and saving to file\n');
    
    % read in all the scan files and metadata
    metadata = get_metadata(opts.radarPath);

    % labels
    fid = fopen(fullfile(opts.radarPath, 'labels.csv'));
    labeldata = textscan(fid, strtrim(repmat('%s ', 1, 3)), ...
                                    'Delimiter', ',', 'HeaderLines', 1);
    metadata.label = get_scan_labels(metadata.scanID, labeldata); 

    % render the scans and save 
    imdb = get_imdb_rendered_scans(metadata, opts);
    save(fullfile(opts.renderPath, 'metadata.mat'), '-struct', 'metadata');
    save(fullfile(opts.renderPath, 'imdb.mat'), '-struct', 'imdb');
else
    fprintf('Reading in pre-computed metadata and imdb from disk\n');
    metadata = load(fullfile(opts.renderPath, 'metadata.mat'));
    imdb = load(fullfile(opts.renderPath, 'imdb.mat'));
end

% remove paths to avoid VL_FEAT version conflict
rmpath(genpath(fullfile(RADAR_PATH, 'rclass2014')));
rmpath(genpath(fullfile(RADAR_PATH, 'rscancv')));
rmpath(genpath(fullfile(RADAR_PATH, 'wsrlib')));


% -------------------------------------------------------------------------
function imdb = get_imdb_rendered_scans(metadata, opts)
% ------------------------------------------------------------------------- 
names = {};
labels = [];
stations = {};
k = 0;

% render the scans and save to file
for i = 1:length(metadata.filelist)
    outDir = fullfile(opts.renderPath, metadata.station{i});
    vl_xmkdir(outDir);
    [a,b,c] = fileparts(metadata.filelist{i});
    swp = strjoin(strsplit(num2str(opts.sweep),' '), '_');
    if isequal(opts.mode, 'dz')
        imgName = [b '_' swp '.jpg'];
    else
        imgName = [b '_' num2str(opts.mode) '_' ...
                    swp '.jpg'];
    end
    outPath = fullfile(outDir, imgName);
    
    % doesn't overwrite if output file already exists
    if ~exist(outPath, 'file')
        try
            switch opts.mode
                case 'dz'
                    img = renderscan(metadata.filelist{i}, metadata.station{i}, ...
                                     'dz', 'sweeps', opts.sweep);
                case 'vr'
                    img = renderscan(metadata.filelist{i}, metadata.station{i}, ...
                                     'vr', 'sweeps', opts.sweep);
                case 'sw'
                    img = renderscan(metadata.filelist{i}, metadata.station{i}, ...
                                     'sw', 'sweeps', opts.sweep);
                otherwise
                    error('Incorrect option for opts.modality');
            end
            imwrite(img, outPath);
            fprintf('%d. Saving render to disk: %s\n', i, outPath);
        catch
            warning('Error in renderscan() while rendering %s\n', ...
                            metadata.filelist{i});
            continue;
        end      
    else
        fprintf('%d. File already exists on disk: %s\n', i, outPath);
    end
    names{end+1} = [metadata.station{i} '/' imgName];
    labels(end+1) = double(metadata.label(i));
    stations{end+1} = metadata.station{i};
end

% imdb structure
imdb.images.name = names;
labels(labels==0) = 2;  % make lables 1,2 from 0,1
imdb.images.label = labels;
imdb.images.station = stations;
imdb.images.id = [1:length(imdb.images.name)];

imdb.imageDir = opts.renderPath;
imdb.classes.name = {'no-rain', 'rain'};
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;

% No standard image splits are provided for this dataset
imdb.sets = {'train', 'val', 'test'};
imdb.images.set = zeros(1,length(imdb.images.id));
for c = 1:length(imdb.classes.name), 
    isclass = find(imdb.images.label == c);
    
    % split equally into train, val, test
    order = randperm(length(isclass));
    subsetSize = ceil(length(order)/3);
    train = isclass(order(1:subsetSize));
    val = isclass(order(subsetSize+1:2*subsetSize));
    test  = isclass(order(2*subsetSize+1:end));
    
    imdb.images.set(train) = 1;
    imdb.images.set(val) = 2;
    imdb.images.set(test) = 3;
end


                                        
% -------------------------------------------------------------------------
function metadata = get_metadata(radarPath)
% ------------------------------------------------------------------------- 
                                       
b = dir(radarPath);
b = b(arrayfun(@(x) ~strcmp(x.name(1),'.'),b)); %remove hidden files

% folders - 'KBGM', 'KOKX', ...
folders = b(arrayfun(@(x) (x.isdir), b));

filelist = {};
scan_id = [];
station = {};

for ii = 1:numel(folders)
    b = dir(fullfile(radarPath,folders(ii).name));
    b = b(arrayfun(@(x) ~strcmp(x.name(1),'.'),b));
    subfolders = b(arrayfun(@(x) (x.isdir), b));
    
    for jj = 1:numel(subfolders)
        subfolderPath = fullfile(radarPath,folders(ii).name, subfolders(jj).name);         
        fid = fopen(fullfile(subfolderPath, 'meta.csv'), 'r');
        meta = textscan(fid, strtrim(repmat('%s ', 1, 11)), ...
                                'Delimiter', ',', 'HeaderLines', 1);
        
        a = cellfun(@(x)(fullfile(subfolderPath, x)), meta{9}, 'un', 0);                    
        filelist = vertcat(filelist, a);             
        station = vertcat(station, meta{2});       
        b = cellfun(@str2num, meta{1}, 'un', 0);
        scan_id = vertcat(scan_id, cell2mat(b));
    end
end  
metadata.filelist = filelist;
metadata.scanID = scan_id;
metadata.station = station;
                                        
                                        
                                        
% -------------------------------------------------------------------------
function labels = get_scan_labels(scanID, metadata)
% -------------------------------------------------------------------------
if iscell(scanID), scanID = cell2mat(scanID); end

labeledScans = (cellfun(@(x)(str2double(x)), metadata{1}, 'un', 0))';
labeledScans = cell2mat(labeledScans);

[~,labelIndex] = ismember(scanID, labeledScans); % match data to labels

labelA = metadata{2}; labelA = labelA(labelIndex);
labelB = metadata{3}; labelB = labelB(labelIndex);

% labels where both annotaters agree are '1', rest '0'
matchingLabels = cellfun(@(x,y)(isequal(x,y)), labelA, labelB, 'un', 0); 
matchingLabels = cell2mat(matchingLabels);

labels = cellfun(@(x)(isequal(x,'accept')), labelA); 
labels = labels & matchingLabels;



% -------------------------------------------------------------------------
function [ gif ] = renderscan( path, callsign, mode, varargin )
% -------------------------------------------------------------------------
%RENDER_SCAN Copy of original RENDERSCAN, but supports multiple modalities
%such as radial velocity ('vr') and spectrum width ('sw') in addition to
%the default relfectivity ('dz').

%radar loading params
DEFAULT_COORDINATES     = 'cartesian'; % 'polar', 'cartesian'
DEFAULT_SWEEPS          = 1;           %a vector of length 1, 2, or 3

%radar alignment params
DEFAULT_AZRES           = 0.5;         %positive real
DEFAULT_RANGERES        = 250;         %positive real
DEFAULT_RANGEMAX        = 37500;       %positive real

%image dimension params
DEFAULT_DIM             = uint16(800); %positive int
DEFAULT_DZLIM           = [-5, 30];    %tuple of reals
DEFAULT_DZMAP           = gray(256);   %colormap

DEFAULT_WRITE_LOCATION  = [];

% parse inputs
parser = inputParser;
addRequired  (parser,'path',                                @(x) exist(x,'file') == 2);
addRequired  (parser,'callsign',                            @(x) validateattributes(x,{'char'},{'numel',4}));
addParamValue(parser,'coordinates', DEFAULT_COORDINATES,    @(x) any(validatestring(x,{'polar','cartesian'})));
addParamValue(parser,'sweeps',      DEFAULT_SWEEPS,         @(x) validateattributes(x,{'numeric'},{'nonempty','positive','integer'}));% && numel(x) <= 3);
addParamValue(parser,'azres',       DEFAULT_AZRES,          @(x) validateattributes(x,{'numeric'},{'>',0}));
addParamValue(parser,'rangeres',    DEFAULT_RANGERES,       @(x) validateattributes(x,{'numeric'},{'>',0}));
addParamValue(parser,'rangemax',    DEFAULT_RANGEMAX,       @(x) validateattributes(x,{'numeric'},{'>',0}));
addParamValue(parser,'dim',         DEFAULT_DIM,            @(x) validateattributes(x,{'numeric'},{'integer','>',0}));
addParamValue(parser,'dzlim',       DEFAULT_DZLIM,          @(x) validateattributes(x,{'numeric'},{'numel',2}));
addParamValue(parser,'dzmap',       DEFAULT_DZMAP,          @(x) iptcheckmap(x,'renderscan','dzmap'));
addParamValue(parser,'outpath',     DEFAULT_WRITE_LOCATION, @(x) isempty(x) || exist(fileparts(x), 'file') == 7);

parse(parser, path, callsign, varargin{:});

path        = parser.Results.path;
callsign    = parser.Results.callsign;
coordinates = parser.Results.coordinates;
sweeps      = parser.Results.sweeps;
azres       = parser.Results.azres;
rangeres    = parser.Results.rangeres;
rangemax    = parser.Results.rangemax;
dim         = parser.Results.dim;
dzlim       = parser.Results.dzlim;
dzmap       = parser.Results.dzmap;
outpath     = parser.Results.outpath;

% read radar

%construct rsl2mat param struct
rsl2mat_params = struct('cartesian', false);%, 'nsweeps', max(sweeps));

radar = rsl2mat(path, callsign, rsl2mat_params);
radar = align_scan(radar, azres, rangeres, rangemax);

% render image

%extract the sweeps as matrices
[data, range, az, ~] = radar2mat(radar, {mode});

%subset the selected sweeps
data = data{1}(:,:,sweeps);

%set NaN to min reflectivity
data(isnan(data)) = min(dzlim);

%allocate gif size
if strcmp(coordinates, 'cartesian')
    %use dim instead
    gif = zeros(dim, dim, size(data,3));
else
    gif = zeros(size(data));
end

%render each sweep into a grayscale gif
for isweep = 1:numel(sweeps)
    sweep = data(:,:,isweep);
    if strcmp(coordinates, 'cartesian')
        %convert to cartesian coordinates
        sweep = mat2cart(sweep, az, range, dim, max(range), 'nearest');
    end
    
    gif(:,:,isweep) = mat2ind(sweep, dzlim, dzmap);
end

%set NaNs in the gif to 0
gif(isnan(gif)) = 0;

%convert to integer type
gif = uint8(gif);

%write image to file (if an outpath exists)
if ~isempty(outpath)
	imwrite(gif, outpath);
end

