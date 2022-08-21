% height = 1;
% width = 1;
% method = 'bicubic';
% src  = '';
% dst  = '';

image = imread(src);
image = imresize(image, [height, width], method);
imwrite(image, dst, 'png');