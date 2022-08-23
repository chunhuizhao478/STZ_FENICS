%post-process stz locs distribution
intv = 1;
staptr = 1;
endptr = 223;
time = 0;
dt = 1e-3;

findreactindex_arr = [];

%save png files
for i = staptr : intv : endptr 

    time = time + dt;

    stzlocs = readmatrix("data/mid_data"+string(i)+".txt");

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    findendreactindex = find(stzlocs(:,3) == 3);
    findreactindex_arr = cat(1,findreactindex_arr,findendreactindex);
    findreactindex_arr = unique(findreactindex_arr, "sorted");

    findnotreactindex = find(stzlocs(:,3) == 0);
    commonelems = intersect(findreactindex_arr, findnotreactindex, 'sorted');

    stzlocs(commonelems,3) = 4;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
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

    index4 = find(stzlocs(:,3) == 4);
    index4_locs = stzlocs(index4,1:2);
    index4_n = size(index4_locs, 1);
    
    fig1 = figure("Visible","off");
    n = size(stzlocs,1);
    viscircles(index3_locs,ones(index3_n,1)*0.6,'Color','b'); hold on; %end active
    viscircles(index2_locs,ones(index2_n,1)*0.6,'Color','m'); hold on; %at threshold
    viscircles(index1_locs,ones(index1_n,1)*0.6,'Color','r'); hold on; %current active
    viscircles(index0_locs,ones(index0_n,1)*0.6,'Color','k'); hold on; %no active
    viscircles(index4_locs,ones(index4_n,1)*0.6,'Color','Y'); hold on; %no active, wait for reactive
    set(fig1,"Position",[64,256,500,500]);
    view(2)
    title("STZ Distribution : T = "+string(time))
    xlabel("x");
    ylabel("y");
    axis equal
    xline(0); xline(64);
    yline(0); yline(256);
    xlim([0,64]);
    ylim([0,256]);

    saveas(fig1,"imagfolder/imag"+string(i)+".png");

end

%make video
% writerObj = VideoWriter('imagfolder/imagvideo.mp4');
% writerObj.FrameRate = 1;
% open(writerObj);
% for K = staptr : intv : endptr 
%   filename = sprintf("imagfolder/imag"+"%d.png", K);
%   thisimage = imread(filename);
%   drawnow;
%   writeVideo(writerObj, thisimage);
% end
% close(writerObj);
