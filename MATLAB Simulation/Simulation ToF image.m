%Read image
Image = imread("Enter location"); %change to location of image
whos Image
new = Image(:,:,1);
new2 = Image(:,:,2);
new3 = Image(:,:,3);
figure;
subplot(2,2,1);
imshow(Image)
title("ToF image")

% Step 1: Read the image
imageRGB = Image;
% Step 2: Convert the image from RGB to HSV
imageHSV = rgb2hsv(imageRGB);
whos imageHSV
% Step 3: Extract the Hue channel
hueChannel = imageHSV(:,:,1)

% Display the original image and the hue channel

subplot(2,2,2)
imshow(hueChannel);
title('Hue Channel');



%make Submatrix and find the sum of each submatrix
[subMatrices,sumMatrix] = sub_matrix(hueChannel);

disp("image matrix")
disp(sumMatrix)

%convert to 0 - 255
[row,col] = size(subMatrices{1,1});
% Define the original range
max_new = row*col;
max_old = 255;

% Sample original values (replace with your actual data)
original_values = sumMatrix;

% Apply the linear transformation
new_values_old = (original_values / max_old) * max_new;
new_values = abs(new_values_old - 255);

% Display the results
disp('Original values:');
disp(original_values);
disp('Mapped values:');
disp(new_values);

%Display
subplot(2,2,3);
b = bar3(new_values);
title("Result")
colorbar;
colormap("bone");

for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
subplot(2,2,4)
surf(new_values);

view([135 85])


