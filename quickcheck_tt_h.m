clear all; close all; clc;
stzlocs = readmatrix("data/mid_data1900.txt");

index0 = find(stzlocs(:,3) == 0);
index0_locs = stzlocs(index0,1:2);
index0_n = size(index0_locs, 1);

index1 = find(stzlocs(:,3) == 1);
index1_locs = stzlocs(index1,1:2);
index1_n = size(index1_locs, 1);

index2 = find(stzlocs(:,3) == 2);
index2_locs = stzlocs(index2,1:2);
index2_n = size(index2_locs, 1);

index3 = find(stzlocs(:,3) == 3);
index3_locs = stzlocs(index3,1:2);
index3_n = size(index3_locs, 1);

figure(1);
n = size(stzlocs,1);
viscircles(index3_locs,ones(index3_n,1)*0.6,'Color','b'); hold on; %end active b
viscircles(index2_locs,ones(index2_n,1)*0.6,'Color','b'); hold on; %at threshold m
viscircles(index1_locs,ones(index1_n,1)*0.6,'Color','b'); hold on; %current active r
%viscircles(index0_locs,ones(index0_n,1)*0.6,'Color','b'); hold on; %no active k
%viscircles([128,32],0.6,'Color','r'); hold on;
axis equal
title('Spatial Distribution of Multiple STZs Growth')
xlim([0,64]);
ylim([0,256]);


curve = readmatrix("curve/mid_curve.txt");
figure(2);
plot(curve(:,1),curve(:,2));
title("shear stress vs shear strain curve");
xlabel("shear strain");
ylabel("shear stress");
xlim([0,0.03]);
ylim([0,2]);

% figure(3);
% engc = readmatrix("engcsave.txt");
% histogram(engc);
% 
% figure(4);
% stzloc = readmatrix("stzlocs.txt");
% sizestzloc = size(stzloc, 1);
% viscircles(stzloc,ones(sizestzloc,1)*0.6,'Color','b');
% axis equal
