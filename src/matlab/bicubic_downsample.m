% scale_numerator = 1;
% scale_denominator = 1;
% folder_src  = '';
% folder_des_hr = '';
% folder_des_lr = '';
extensions  =  {'*.png'};

hr_files   =  [];
for i = 1:length(extensions)
    hr_files = cat(1, hr_files, dir(fullfile(folder_src, extensions{i})));
end
n_imgs = length(hr_files);

%% generate LR images
for idx_im = 1:n_imgs
    fprintf('Processing image #%d\n', idx_im);
    hr_image = imread(fullfile(folder_src, hr_files(idx_im).name));
    
    % crop center
    [height, width, channels]=size(hr_image);
    hr_height = floor(height/scale_numerator) * scale_numerator;
    hr_width = floor(width/scale_numerator) * scale_numerator;
    r = [floor(width/2)-floor(hr_width/2),floor(height/2)-floor(height/2), hr_width, hr_height];
    hr_image = imcrop(hr_image, r);

    % bicubic downsampling
    lr_height = hr_height/scale_numerator*scale_denominator;
    lr_width = hr_width/scale_numerator*scale_denominator;
    lr_image = imresize(hr_image, [lr_height, lr_width], 'bicubic');
        
    % save image
    hr_image_name = fullfile(folder_des_hr, hr_files(idx_im).name);
    imwrite(hr_image, hr_image_name, 'png');
    
    lr_image_name = fullfile(folder_des_lr, hr_files(idx_im).name);
    imwrite(lr_image, lr_image_name, 'png');
end


