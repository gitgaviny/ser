clear all  % 清除工作空间的所有变量，函数，和MEX文件
wavpath = 'D:\data\iemocap\session4\';
cd(wavpath);                        
filelist = dir('*.wav');  
filelist = struct2cell(filelist);   
filelist = filelist(1,:)';  %最终保存n*1的所有带后缀的文件名cell
filename = cell(length(filelist),1);  %cell数组一般被叫做元胞数组，它的每个单元可以储存不同的数据类型
w=256;   %窗函数，x将被分成w段 
n=256;    %计算傅里叶变换的点数  
ov=w/2;   %各段之间重叠的采样点数

for i=1:length(filelist)
    a = filelist(i);  %a是所有文件的文件名
    [pathstr,name,ext] = fileparts(a{1});  %获取文件名的组成部分 路径名，文件名，扩展名
    filename{i,1} = name;  %filename保存不带后缀的文件名
end 
lenS4=[];
for i=1:length(filelist)
    disp(filename{i,1});
    wavfile = [wavpath,filename{i},ext];
    [x,fs] = audioread(wavfile);   
    lenx = length(x(:));             
    x_start = 1;
    x_end = x_start + lenx - 1;
    t = x(x_start:x_end,:);  %t保存这一分段下所有采样点数据
    yy(:,:) = t;         
    xx=double(yy(:)');  
    [S,~,~,~]=spectrogram(xx,w,ov,n,fs); %功能：使用短时傅里叶变换得到信号的频谱图
    S=log(1+abs(S));
    lens = length(S(1,:,:));
    if lens > 700
       lens = 700;
       S(:,701:length(S(1,:)))=[];
    elseif lens < 700
       S(:,length(S(1,:))+1:700)=0;
    end
    lenS4 = [lenS4,lens];
    z(1,:,:)=S';   %z保存音谱图
    imse4{i,:}=z;
    clear x_end L yy xx z;
end
a=length(imse4);                     %xiugai 
A4=imse4{1,1};                       %xiugai 
for j=2:a
    fprintf('xie=%d\n',j);
    A4=[A4;imse4{j,1}];                %xiugai 
end
lenS4 = lenS4';
load F:\singlecorpus\data\labels\imse4label
load F:\singlecorpus\data\labels\gender4label
y4=imse4label;    
g4=gender4label;
save F:\singlecorpus\data\imse_4.mat A4 y4 g4 lenS4 -v7.3