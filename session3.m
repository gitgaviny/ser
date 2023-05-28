clear all  % ��������ռ�����б�������������MEX�ļ�
wavpath = 'D:\data\iemocap\session3\';
cd(wavpath);                        
filelist = dir('*.wav');  
filelist = struct2cell(filelist);   
filelist = filelist(1,:)';  %���ձ���n*1�����д���׺���ļ���cell
filename = cell(length(filelist),1);  %cell����һ�㱻����Ԫ�����飬����ÿ����Ԫ���Դ��治ͬ����������
w=256;   %��������x�����ֳ�w�� 
n=256;    %���㸵��Ҷ�任�ĵ���  
ov=w/2;   %����֮���ص��Ĳ�������

for i=1:length(filelist)
    a = filelist(i);  %a�������ļ����ļ���
    [pathstr,name,ext] = fileparts(a{1});  %��ȡ�ļ�������ɲ��� ·�������ļ�������չ��
    filename{i,1} = name;  %filename���治����׺���ļ���
end 
lenS3=[];
for i=1:length(filelist)
    disp(filename{i,1});
    wavfile = [wavpath,filename{i},ext];
    [x,fs] = audioread(wavfile);   
    lenx = length(x(:));             
    x_start = 1;
    x_end = x_start + lenx - 1;
    t = x(x_start:x_end,:);  %t������һ�ֶ������в���������
    yy(:,:) = t;         
    xx=double(yy(:)');  
    [S,~,~,~]=spectrogram(xx,w,ov,n,fs); %���ܣ�ʹ�ö�ʱ����Ҷ�任�õ��źŵ�Ƶ��ͼ
    S=log(1+abs(S));
    lens = length(S(1,:,:));
    if lens > 700
       lens = 700;
       S(:,701:length(S(1,:)))=[];
    elseif lens < 700
       S(:,length(S(1,:))+1:700)=0;
    end
    lenS3 = [lenS3,lens];
    z(1,:,:)=S';   %z��������ͼ
    imse3{i,:}=z;
    clear x_end L yy xx z;
end
a=length(imse3);                     %xiugai 
A3=imse3{1,1};                       %xiugai 
for j=2:a
    fprintf('xie=%d\n',j);
    A3=[A3;imse3{j,1}];                %xiugai 
end
lenS3 = lenS3';
load F:\singlecorpus\data\labels\imse3label
load F:\singlecorpus\data\labels\gender3label
y3=imse3label;    
g3=gender3label;
save F:\singlecorpus\data\imse_3.mat A3 y3 g3 lenS3 -v7.3