%make video
writerObj = VideoWriter('imagfolder/imagvideo.avi');
writerObj.FrameRate = 1;
open(writerObj);
for K = staptr : intv : endptr 
  filename = sprintf("imagfolder/imag"+"%d.png", K);
  thisimage = imread(filename);
  drawnow;
  writeVideo(writerObj, thisimage);
end
close(writerObj);