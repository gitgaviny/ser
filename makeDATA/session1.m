wavpath = 'D:\data\iemocap\session1\';
cd(wavpath);                        
filelist = dir('*.wav');  
filelist = struct2cell(filelist);   
filelist = filelist(1,:)';  
filename = cell(length(filelist),1); 
w=256;   
n=256;      
ov=w/2;  

for i=1:length(filelist)
    a = filelist(i); 
    [pathstr,name,ext] = fileparts(a{1});  
    filename{i,1} = name;  
end 
lenS1=[];
for i=1:length(filelist)
    disp(filename{i,1});
    wavfile = [wavpath,filename{i},ext];
    [x,fs] = audioread(wavfile);   
    lenx = length(x(:));             
    x_start = 1;
    x_end = x_start + lenx - 1;
    t = x(x_start:x_end,:);  
    yy(:,:) = t;         
    xx=double(yy(:)');  
    [S,~,~,~]=spectrogram(xx,w,ov,n,fs); 
    S=log(1+abs(S));
    lens = length(S(1,:,:));
    if lens > 700
       lens = 700;
       S(:,701:length(S(1,:)))=[];
    elseif lens < 700
       S(:,length(S(1,:))+1:700)=0;
    end
    lenS1 = [lenS1,lens];
    z(1,:,:)=S';  
    imse1{i,:}=z;
    clear x_end L yy xx z;
end
a=length(imse1);                     
A1=imse1{1,1};                      
for j=2:a
    fprintf('xie=%d\n',j);
    A1=[A1;imse1{j,1}];                
end
lenS1 = lenS1';
load F:\singlecorpus\data\labels\imse1label
load F:\singlecorpus\data\labels\gender1label
y1=imse1label;  
g1=gender1label;
% save F:\singlecorpus\data\imse_1.mat A1 y1 g1 lenS1 -v7.3
