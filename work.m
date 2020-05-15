function varargout = work(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @work_OpeningFcn, ...
    'gui_OutputFcn',  @work_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function work_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;

guidata(hObject, handles);
clear all;
addpath('./function');


function varargout = work_OutputFcn(hObject, eventdata, handles)

varargout{1} = handles.output;


function start_Callback(hObject, eventdata, handles)


function open_Callback(hObject, eventdata, handles)
%ѡȡȫ�ֱ�����¼ͼ������ļ������ļ�·��
global image filename filepath;
%ѡȡͼƬ����¼�ļ������ļ�·��
[filename,filepath]=uigetfile({'*.*'},'ѡ��ͼƬ','./image');
%�ж��Ƿ�ѡȡ��ͼƬ
if filename~=0
    %��ʾ�ļ����ڴ�����
    set(handles.filename,'String',filename);
    %�趨��ʾ����
    axes(handles.image);
    %�����ļ������ļ�·����ȡͼƬ
    image=imread([filepath,filename]);
    %��ʾͼƬ
    imshow(image);
end


function popupmenu1_Callback(hObject, eventdata, handles)


function popupmenu1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function filename_CreateFcn(hObject, eventdata, handles)


function first_Callback(hObject, eventdata, handles)



%��
function add_Callback(hObject, eventdata, handles)

%ȫ�ֱ�������ȡ�Զ�ȡ��ͼƬ
global image
if isempty(image)~=1
    %ѡ����Ҫ���ϵ�ͼƬ
    [filename1,filepath1]=uigetfile({'*.*'},'ѡ��ͼƬ');
    if filename1~=0
        %��ȡ��Ҫ���ϵ�ͼƬ
        image1=imread([filepath1,filename1]);
        %����һ���´��ڣ�����Ϊ���ӡ�������ʾtoolbar��menubar���ţ�����һ��һ�����еı���ڵ�һ����ʾԭͼƬ������Ϊ�����ϵ�ͼƬ��
        figure('name','��','toolbar','none','menubar','none','NumberTitle','off');subplot(1,2,1);imshow(image1);title('���ϵ�ͼƬ');
        %��ȡԭͼƬ�Ĵ�С
        [width,length,height]=size(image);
        %��ʼ������
        res=zeros(width,length,height);
        %double����ת����uint8
        res=uint8(res);
        %��ֹ���ϵ�ͼƬ��ԭͼ��С��һ�£����в���
        image1=imresize(image1,[width length]);
        
        %��ͼ���
        for i=1:width
            for j=1:length
                for k=1:height
                    res(i,j,k)=image(i,j,k)+image1(i,j,k);
                end
            end
        end
        %�ڵڶ�����ʾͼƬ������Ϊ�����ϵĽ����
        subplot(1,2,2);imshow(res);title('���Ϻ�Ľ��');
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end

%��
%��Ӵ�����ͬ��ע�Ͳο���
function div_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [filename1,filepath1]=uigetfile({'*.*'},'ѡ��ͼƬ');
    if filename1~=0
        image1=imread([filepath1,filename1]);
        figure('name','��','toolbar','none','menubar','none','NumberTitle','off');subplot(1,2,1);imshow(image1);title('��ȡ��ͼƬ');
        
        [width,length,height]=size(image);
        res=zeros(width,length,height);
        res=uint8(res);
        image1=imresize(image1,[width length]);
        
        for i=1:width
            for j=1:length
                for k=1:height
                    res(i,j,k)=image(i,j,k)-image1(i,j,k);
                end
            end
        end
        subplot(1,2,2);imshow(res);title('��ȥ��Ľ��');
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end



% �ƶ�
function move_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    %����һ������Ի��򣬻�ȡ�ƶ��ķ���
    %�����Ͻ�Ϊԭ�㣬����Ϊx������������Ϊy��������
    
    %��ʾ�ַ���
    prompt={'x��','y��'};
    %�Ի�������
    name='�ƶ�����';
    %��ʾ������
    numlines=1;
    %Ĭ����ֵ
    defaultanswer={'50','50'};
    
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    
    answer=inputdlg(prompt,name,numlines,defaultanswer,options);
    if isempty(answer)~=1
        %��ȡ����ֵ������ת��������ƶ���������ֵ
        delY=answer(1);
        delX=answer(2);
        delX=cell2mat(delX);
        delY=cell2mat(delY);
        delX=str2num(delX);
        delY=str2num(delY);
        
        
        [width,length,height]=size(image);
        res=zeros(width,length,height);
        res=uint8(res);
        % ƽ��
        tras = [1 0 delX; 0 1 delY; 0 0 1]; % ƽ�Ƶı任����
        for i = 1 : width
            for j = 1 : length
                temp = [i; j; 1];
                temp = tras * temp; % ����˷�
                x = temp(1, 1);
                y = temp(2, 1);
                % �任���λ���ж��Ƿ�Խ��
                if (x <= width) & (y <= length) & (x >= 1) & (y >= 1)
                    res(x, y,:) = image(i, j,:);
                end
            end
        end
        figure('name','�ƶ�','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('�ƶ����ͼƬ');
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end

%����
function mirror_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [width,length,height]=size(image);
    for i = 1 : width
        for j = 1 : length
            x = i;
            y = length - j + 1;
            res(x, y,:) = image(i, j,:);
        end
    end
    figure('name','����','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('������ͼƬ');
else
    warndlg('��ѡ��ͼƬ','Waring');
end



% ��ת
function rotate_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    %��ʾ�ַ���
    prompt={'��ת�Ƕ�'};
    %�Ի�������
    name='��ת';
    %��ʾ������
    numlines=1;
    %Ĭ����ֵ
    defaultanswer={'30'};
    
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    
    answer=inputdlg(prompt,name,numlines,defaultanswer,options);
    if isempty(answer)~=1
        %��ȡ����ֵ������ת���������ת�Ƕ�
        ang=answer(1);
        ang=cell2mat(ang);
        ang=str2num(ang);
        
        res=imrotate(image,ang);
        figure('name','��ת','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('��ת���ͼƬ');
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end



function second_Callback(hObject, eventdata, handles)


%���ȱ任
function bright_trans_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    res=image;
    %     image_size=size(image);
    %     dimension=numel(image_size);
    %     if dimension==3
    %         res=rgb2gray(image);
    %     end
    
    %��ʾ�ַ���
    prompt={'low in  high in������֮���ÿո������','low out high out������֮���ÿո������','gamma'};
    %�Ի�������
    name='���ȱ任';
    %��ʾ������
    numlines=1;
    %Ĭ����ֵ
    defaultanswer={'0.3 0.8','0 1','1'};
    
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    
    answer=inputdlg(prompt,name,numlines,defaultanswer,options);
    if isempty(answer)~=1
        %��ȡ����ֵ������ת�������low_in,high_in,low_out,high_out,gamma
        in=answer(1);
        out=answer(2);
        gam=answer(3);
        out=cell2mat(out);
        in=cell2mat(in);
        gam=cell2mat(gam);
        out=str2num(out);
        in=str2num(in);
        gam=str2num(gam);
        
        figure('name','ֱ��ͼ','toolbar','none','menubar','none','NumberTitle','off');
        subplot(1,2,1);imhist(res);title('ԭͼ��ֱ��ͼ');
        res=imadjust(res,in,out,gam);%�������ȱ任
        subplot(1,2,2);imhist(res);title('���ȱ任���ֱ��ͼ');
        figure('name','���ȱ任','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('���ȱ任���ͼƬ');
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end


% ֱ��ͼ����
function Histogram_Equalization_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    res=histeq(image); %��ԭͼ�����ֱ��ͼ���⻯����
    figure('name','ֱ��ͼ����','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('ֱ��ͼ������ͼƬ');  %��ԭͼ�������Ļ����;��ʾֱ��ͼ���⻯���ͼ��
    
    figure('name','ֱ��ͼ','toolbar','none','menubar','none','NumberTitle','off');
    %��ֱ��ͼ���⻯���ͼ�������Ļ����;��һ����ͼ��Ϊ��������ͼ�ĵ�1��ͼ,��ԭͼ��ֱ��ͼ��ʾΪ256���Ҷ�,��ԭͼ��ֱ��ͼ�ӱ�����
    subplot(1,2,1) ;imhist(image,256);  title('ԭͼ��ֱ��ͼ') ;
    %����2����ͼ,�����⻯��ͼ���ֱ��ͼ��ʾΪ256���Ҷ�,�����⻯��ͼ��ֱ��ͼ�ӱ�����
    subplot(1,2,2);  imhist(res,256) ; title('����任���ֱ��ͼ') ;
else
    warndlg('��ѡ��ͼƬ','Waring');
end


% laplacian���ӿ����˲�
function Spatial_filtering_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    w4=fspecial('laplacian',0);
    res=image;
    res=im2double(res);
    g4=res-imfilter(res,w4,'replicate');
    figure('name','laplacian���ӿ����˲�','toolbar','none','menubar','none','NumberTitle','off');imshow(g4);title('����Ϊ-4������˹��Ч��');
    
else
    warndlg('��ѡ��ͼƬ','Waring');
end


%��ά����Ҷ�任
function Fourier_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    image_size=size(image);
    dimension=numel(image_size);
    if dimension==3
        res=rgb2gray(image);
    else
        res=image;
    end
    fftI=fft2(res);       %��ά��ɢ����Ҷ�任
    sfftI=fftshift(fftI);  %ֱ�������Ƶ�Ƶ������
    RR=real(sfftI);    %ȡ����Ҷ�任��ʵ��
    II=imag(sfftI);     %ȡ����Ҷ�任���鲿
    A=sqrt(RR.^2+II.^2); %����Ƶ�׷�ֵ
    A=(A-min(min(A)))/(max(max(A))-min(min(A)))*225; %��һ��
    figure('name','����Ҷ�任','toolbar','none','menubar','none','NumberTitle','off');imshow(A);title('ͼ���Ƶ��');
else
    warndlg('��ѡ��ͼƬ','Waring');
end


% ������˹�˲���
function Butterworth_filters_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [sel,ok]=listdlg('liststring',{'��ͨ','��ͨ'},...
        'listsize',[180 80],'OkString','ȷ��','CancelString','ȡ��',...
        'promptstring','������˹�˲���ѡ��','name','ѡ���˲���','selectionmode','single');
    if ok==1
        
        %��ʾ�ַ���
        prompt={'�������ֹƵ�ʣ�'};
        %�Ի�������
        name='��ֹƵ��';
        %��ʾ������
        numlines=1;
        %Ĭ����ֵ
        defaultanswer={'10'};
        
        options.Resize='on';
        options.WindowStyle='normal';
        options.Interpreter='tex';
        
        answer=inputdlg(prompt,name,numlines,defaultanswer,options);
        if isempty(answer)~=1
            
            d0=answer(1);
            d0=cell2mat(d0);
            d0=str2num(d0);
            
            image_size=size(image);
            dimension=numel(image_size);
            if dimension==3
                res=rgb2gray(image);
            else
                res=image;
            end
            
            figure('toolbar','none','menubar','none','NumberTitle','off');
            J1=imnoise(res,'salt & pepper');                  %����У������
            subplot(2,3,1);imshow(J1);title('���ӽ�������ͼ');
            f=double(J1);           %��������ת����MATLAB��֧��ͼ����޷������͵ļ���
            g=fft2(f);              %����Ҷ�任
            g=fftshift(g);          %ת����������
            RR=real(g);%ȡ����Ҷ�任��ʵ��
            II=imag(g);%ȡ����Ҷ�任���鲿
            A=sqrt(RR.^2+II.^2);%����Ƶ�׷�ֵ
            A=(A-min(min(A)))/(max(max(A))-min(min(A)))*225;%��һ��
            subplot(2,3,2);imshow(A);title('�˲�ǰ��Ƶ��');%��ʾԭͼ���Ƶ��
            h1=g;[M,N]=size(g);nn=2;
            m=fix(M/2);n=fix(N/2);
            
            if sel==1%��ͨ
                
                for i=1:M
                    for j=1:N
                        d=sqrt((i-m)^2+(j-n)^2);
                        h=1/(1+(d/d0)^(2*nn));%���ɶ��װ�����˹��ͨ�˲���
                        h1(i,j)=h;
                    end
                end
                subplot(2,3,3);imshow(h1);title('��ͨ�˲�����ͼ��');%��ʾ�˲�����ͼ��
                
            elseif sel==2%��ͨ
                
                for i=1:M
                    for j=1:N
                        d=sqrt((i-m)^2+(j-n)^2);
                        h=1/(1+(d0/d)^(2*nn));   %�����ͨ�˲������ݺ���
                        h2=0.5+2*h;    %���high-frequency emphasis����a=0.5,b=2.0
                        h1(i,j)=h2;  %����Ƶ��˲�������ԭͼ��
                    end
                end
                subplot(2,3,3);imshow(h1);title('��ͨ�˲�����ͼ��');%��ʾ�˲�����ͼ��
                
            end
            
            result=h1.*g;          %�˲�����
            RR1=real(result);      %ȡ����Ҷ�任��ʵ��
            II1=imag(result);      %ȡ����Ҷ�任���鲿
            A1=sqrt(RR1.^2+II1.^2);%����Ƶ�׷�ֵ
            A1=(A1-min(min(A1)))/(max(max(A1))-min(min(A1)))*225;%��һ��
            subplot(2,3,4);imshow(A1);title('�˲����Ƶ��'); %��ʾ�˲����Ƶ��
            result=ifftshift(result);%�˲�����
            J2=ifft2(result);%����Ҷ���任
            J3=uint8(real(J2));%ȡʵ��
            subplot(2,3,5);imshow(J3);title('�˲����ͼ��');%��ʾ�˲�Ч��
        end
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end

function thrid_Callback(hObject, eventdata, handles)

% ��ֵ�˲�
function mid_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [sel,ok]=listdlg('liststring',{'��������','��˹����','��������','�˷�����'},...
        'listsize',[180 80],'OkString','ȷ��','CancelString','ȡ��',...
        'promptstring','ѡ����ӵ�����','name','����ѡ��','selectionmode','single');
    if ok==1
        
        image_size=size(image);
        dimension=numel(image_size);
        if dimension==3
            res=rgb2gray(image);
        else
            res=image;
        end
        
        options.Resize='on';
        options.WindowStyle='normal';
        options.Interpreter='tex';
        
        if sel==1       %��������
            %��ʾ�ַ���
            prompt={'�����뽷���������ܶȣ�'};
            %�Ի�������
            name='��������';
            %��ʾ������
            numlines=1;
            %Ĭ����ֵ
            defaultanswer={'0.05'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                m=answer(1);
                m=cell2mat(m);
                m=str2double(m);
                J2=imnoise(res,'salt & pepper',m);
                
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('��������');
                I_Filter1=medfilt2(J2,[3 3]);%���ڴ�СΪ3*3
                subplot(1,4,2);imshow(I_Filter1);title('3*3��ֵ�˲�');
                I_Filter2=medfilt2(J2,[5 5]);%���ڴ�СΪ5*5
                subplot(1,4,3);imshow(I_Filter2);title('5*5��ֵ�˲�');
                I_Filter3=medfilt2(J2,[7 7]);%���ڴ�СΪ7*7
                subplot(1,4,4);imshow(I_Filter3);title('7*7��ֵ�˲�');
            end
        elseif sel==2   %��˹����
            %��ʾ�ַ���
            prompt={'�������˹�����ľ�ֵ��','�������˹�����ķ���'};
            %�Ի�������
            name='��˹����';
            %��ʾ������
            numlines=1;
            %Ĭ����ֵ
            defaultanswer={'0','0.01'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                m=answer(1);
                m=cell2mat(m);
                m=str2double(m);
                
                v=answer(2);
                v=cell2mat(v);
                v=str2double(v);
                J2=imnoise(res,'gaussian',m,v);
                
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('��˹����');
                I_Filter1=medfilt2(J2,[3 3]);%���ڴ�СΪ3*3
                subplot(1,4,2);imshow(I_Filter1);title('3*3��ֵ�˲�');
                I_Filter2=medfilt2(J2,[5 5]);%���ڴ�СΪ5*5
                subplot(1,4,3);imshow(I_Filter2);title('5*5��ֵ�˲�');
                I_Filter3=medfilt2(J2,[7 7]);%���ڴ�СΪ7*7
                subplot(1,4,4);imshow(I_Filter3);title('7*7��ֵ�˲�');
            end
        elseif sel==3   %��������
            
            J2=imnoise(res,'poisson');
            figure('toolbar','none','menubar','none','NumberTitle','off');
            subplot(1,4,1);imshow(J2);title('��������');
            I_Filter1=medfilt2(J2,[3 3]);%���ڴ�СΪ3*3
            subplot(1,4,2);imshow(I_Filter1);title('3*3��ֵ�˲�');
            I_Filter2=medfilt2(J2,[5 5]);%���ڴ�СΪ5*5
            subplot(1,4,3);imshow(I_Filter2);title('5*5��ֵ�˲�');
            I_Filter3=medfilt2(J2,[7 7]);%���ڴ�СΪ7*7
            subplot(1,4,4);imshow(I_Filter3);title('7*7��ֵ�˲�');
            
        elseif sel==4   %�˷�����
            %��ʾ�ַ���
            prompt={'������˷������ķ����ֵΪ0����'};
            %�Ի�������
            name='�˷�����';
            %��ʾ������
            numlines=1;
            %Ĭ����ֵ
            defaultanswer={'0.04'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                
                v=answer(1);
                v=cell2mat(v);
                v=str2double(v);
                J2=imnoise(res,'speckle',v);
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('�˷�����');
                I_Filter1=medfilt2(J2,[3 3]);%���ڴ�СΪ3*3
                subplot(1,4,2);imshow(I_Filter1);title('3*3��ֵ�˲�');
                I_Filter2=medfilt2(J2,[5 5]);%���ڴ�СΪ5*5
                subplot(1,4,3);imshow(I_Filter2);title('5*5��ֵ�˲�');
                I_Filter3=medfilt2(J2,[7 7]);%���ڴ�СΪ7*7
                subplot(1,4,4);imshow(I_Filter3);title('7*7��ֵ�˲�');
            end
        end
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end

% ��ֵ�˲�
function average_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [sel,ok]=listdlg('liststring',{'��������','��˹����','��������','�˷�����'},...
        'listsize',[180 80],'OkString','ȷ��','CancelString','ȡ��',...
        'promptstring','ѡ����ӵ�����','name','����ѡ��','selectionmode','single');
    if ok==1
        
        image_size=size(image);
        dimension=numel(image_size);
        if dimension==3
            res=rgb2gray(image);
        else
            res=image;
        end
        
        options.Resize='on';
        options.WindowStyle='normal';
        options.Interpreter='tex';
        
        if sel==1       %��������
            %��ʾ�ַ���
            prompt={'�����뽷���������ܶȣ�'};
            %�Ի�������
            name='��������';
            %��ʾ������
            numlines=1;
            %Ĭ����ֵ
            defaultanswer={'0.05'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                m=answer(1);
                m=cell2mat(m);
                m=str2double(m);
                J2=imnoise(res,'salt & pepper',m);
                
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('��������');
                I_Filter1=filter2(fspecial('average',3),J2)/255;%����3*3�ľ�ֵ�˲�
                subplot(1,4,2);imshow(I_Filter1);title('3*3ģ���ֵ�˲�');
                I_Filter2=filter2(fspecial('average',5),J2)/255;%����5*5�ľ�ֵ�˲�
                subplot(1,4,3);imshow(I_Filter2);title('5*5ģ���ֵ�˲�');
                I_Filter3=filter2(fspecial('average',7),J2)/255;%����7*7�ľ�ֵ�˲�
                subplot(1,4,4);imshow(I_Filter3);title('7*7ģ���ֵ�˲�');
            end
        elseif sel==2   %��˹����
            %��ʾ�ַ���
            prompt={'�������˹�����ľ�ֵ��','�������˹�����ķ���'};
            %�Ի�������
            name='��˹����';
            %��ʾ������
            numlines=1;
            %Ĭ����ֵ
            defaultanswer={'0','0.01'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                m=answer(1);
                m=cell2mat(m);
                m=str2double(m);
                
                v=answer(2);
                v=cell2mat(v);
                v=str2double(v);
                J2=imnoise(res,'gaussian',m,v);
                
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('��˹����');
                I_Filter1=filter2(fspecial('average',3),J2)/255;%����3*3�ľ�ֵ�˲�
                subplot(1,4,2);imshow(I_Filter1);title('3*3ģ���ֵ�˲�');
                I_Filter2=filter2(fspecial('average',5),J2)/255;%����5*5�ľ�ֵ�˲�
                subplot(1,4,3);imshow(I_Filter2);title('5*5ģ���ֵ�˲�');
                I_Filter3=filter2(fspecial('average',7),J2)/255;%����7*7�ľ�ֵ�˲�
                subplot(1,4,4);imshow(I_Filter3);title('7*7ģ���ֵ�˲�');
            end
        elseif sel==3   %��������
            
            J2=imnoise(res,'poisson');
            figure('toolbar','none','menubar','none','NumberTitle','off');
            subplot(1,4,1);imshow(J2);title('��������');
            I_Filter1=filter2(fspecial('average',3),J2)/255;%����3*3�ľ�ֵ�˲�
            subplot(1,4,2);imshow(I_Filter1);title('3*3ģ���ֵ�˲�');
            I_Filter2=filter2(fspecial('average',5),J2)/255;%����5*5�ľ�ֵ�˲�
            subplot(1,4,3);imshow(I_Filter2);title('5*5ģ���ֵ�˲�');
            I_Filter3=filter2(fspecial('average',7),J2)/255;%����7*7�ľ�ֵ�˲�
            subplot(1,4,4);imshow(I_Filter3);title('7*7ģ���ֵ�˲�');
            
        elseif sel==4   %�˷�����
            %��ʾ�ַ���
            prompt={'������˷������ķ����ֵΪ0����'};
            %�Ի�������
            name='�˷�����';
            %��ʾ������
            numlines=1;
            %Ĭ����ֵ
            defaultanswer={'0.04'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                
                v=answer(1);
                v=cell2mat(v);
                v=str2double(v);
                J2=imnoise(res,'speckle',v);
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('�˷�����');
                I_Filter1=filter2(fspecial('average',3),J2)/255;%����3*3�ľ�ֵ�˲�
                subplot(1,4,2);imshow(I_Filter1);title('3*3ģ���ֵ�˲�');
                I_Filter2=filter2(fspecial('average',5),J2)/255;%����5*5�ľ�ֵ�˲�
                subplot(1,4,3);imshow(I_Filter2);title('5*5ģ���ֵ�˲�');
                I_Filter3=filter2(fspecial('average',7),J2)/255;%����7*7�ľ�ֵ�˲�
                subplot(1,4,4);imshow(I_Filter3);title('7*7ģ���ֵ�˲�');
            end
        end
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end

function forth_Callback(hObject, eventdata, handles)



function point_detection_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    image_size=size(image);
    dimension=numel(image_size);
    if dimension==3
        res=rgb2gray(image);
    else
        res=image;
    end
    
    %����
    w = [-1 -1 -1;-1 8 -1;-1 -1 -1];                    % ������ģ
    g = abs(imfilter(double(res),w));
    T = max(g(:));
    g = g>=T;
    figure('toolbar','none','menubar','none','NumberTitle','off');imshow(g);title('����');
    
else
    warndlg('��ѡ��ͼƬ','Waring');
end


function line_detection_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    image_size=size(image);
    dimension=numel(image_size);
    if dimension==3
        res=rgb2gray(image);
    else
        res=image;
    end
    
    [sel,ok]=listdlg('liststring',{'ˮƽģ��','45��ģ��','��ֱģ��','-45��ģ��'},...
        'listsize',[180 80],'OkString','ȷ��','CancelString','ȡ��',...
        'promptstring','ѡ��ģ��','name','ģ��ѡ��','selectionmode','single');
    if ok==1
        %�߼��
        if sel==1
            w = [-1,-1,-1;2,2,2;-1,-1,-1];%ˮƽģ��
            tit='ˮƽģ��';
        elseif sel==2
            w = [-1,-1,2;-1,2,-1;2,-1,-1];%45��ģ��
            tit='45��ģ��';
        elseif sel==3
            w = [-1,2,-1;-1,2,-1;-1,2,-1];%��ֱģ��
            tit='��ֱģ��';
        elseif sel==4
            w = [2 -1 -1;-1 2 -1;-1 -1 2];%-45��ģ��
            tit='-45��ģ��';
        end
        
        g = imfilter(double(res),w);
        g1 = abs(g);                             % ���ͼ�ľ���ֵ
        T = max(g1(:));
        g2 = g1>=T;
        
        figure('toolbar','none','menubar','none','NumberTitle','off');
        subplot(1,2,1);imshow(g1,[]);title(tit);
        subplot(1,2,2);imshow(g2);title('g>=T');
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end

%��Ե���
function edge_detection_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    image_size=size(image);
    dimension=numel(image_size);
    if dimension==3
        res=rgb2gray(image);
    else
        res=image;
    end
    
    answer=questdlg('�Ƿ����������','�������','Yes','No','Yes');
    if strcmp(answer,'Yes')
        res=imnoise(res,'gaussian',0,0.01);
    end
    
    BW1=edge(res,'sobel');
    BW2=edge(res,'roberts');
    BW3=edge(res,'log');
    BW4=edge(res,'canny');
    BW5=edge(res,'prewitt');
    
    figure('toolbar','none','menubar','none','NumberTitle','off');
    subplot(2,3,1);imshow(res);title('ԭͼ��');
    subplot(2,3,2);imshow(BW1);title('sobel�����');
    subplot(2,3,3);imshow(BW2);title('roberts�����');
    subplot(2,3,4);imshow(BW3);title('log�����');
    subplot(2,3,5);imshow(BW4);title('canny�����');
    subplot(2,3,6);imshow(BW5);title('prewitt�����');
    
    
else
    warndlg('��ѡ��ͼƬ','Waring');
end



function regiongrow_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    image_size=size(image);
    dimension=numel(image_size);
    if dimension==3
        res=rgb2gray(image);
    else
        res=image;
    end
    
    [width,length]=size(res);
    
    msgbox('��ѡȡ����λ��');
    uiwait;
    
    %����������
    [x,y] = ginput(1);
    x=fix(x);
    y=fix(y);
    while x>length || y>width || x<0 || y<0
        msgbox('����ͼƬ��ѡȡ����λ��');
        uiwait;
        [x,y] = ginput(1);
        x=fix(x);
        y=fix(y);
    end
    
    res = im2double(res);
    J = regiongrowing(res,x,y,0.2);
    figure('toolbar','none','menubar','none','NumberTitle','off');
    imshow(res+J);
    
else
    warndlg('��ѡ��ͼƬ','Waring');
end



function Split_merge_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    image_size=size(image);
    dimension=numel(image_size);
    if dimension==3
        res=rgb2gray(image);
    else
        res=image;
    end
    %�и�ͼƬ
    [wid,len]=size(res);
    
    if wid~=len
        num=max(wid,len);
        num=fix(log2(num));
        if power(2,num) < 512
            num=512;
        else
            num=power(2,num);
        end
        res=imresize(res,[num num]);
    end
    
    %��ʾ�ַ���
    prompt={'���������ƶȣ�'};
    %�Ի�������
    name='���ƶ�';
    %��ʾ������
    numlines=1;
    %Ĭ����ֵ
    defaultanswer={'0.27'};
    
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    
    answer=inputdlg(prompt,name,numlines,defaultanswer,options);
    if isempty(answer)~=1
        
        v=answer(1);
        v=cell2mat(v);
        v=str2double(v);
        
        S = qtdecomp(res,v);%���ƶ�׼��v
        blocks = repmat(uint8(0),size(S));
        
        for dim = [512 256 128 64 32 16 8 4 2 1]
            numblocks = length(find(S==dim));
            if (numblocks > 0)
                values = repmat(uint8(1),[dim dim numblocks]);
                values(2:dim,2:dim,:) = 0;
                blocks = qtsetblk(blocks,S,dim,values);
            end
        end
        
        blocks(end,1:end) = 1;
        blocks(1:end,end) = 1;
        figure('toolbar','none','menubar','none','NumberTitle','off');
        imshow(blocks,[]);
        
    end
    
else
    warndlg('��ѡ��ͼƬ','Waring');
end


function fifth_Callback(hObject, eventdata, handles)

%DCTƵ��
function DCT_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    image_size=size(image);
    dimension=numel(image_size);
    if dimension==3
        res=rgb2gray(image);
    else
        res=image;
    end
    
    res=im2double(res);
    I=dct2(res);
    figure('toolbar','none','menubar','none','NumberTitle','off');
    imshow(I);
else
    warndlg('��ѡ��ͼƬ','Waring');
end

%ͼ��ѹ��
function Image_Compression_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    image_size=size(image);
    dimension=numel(image_size);
    if dimension==3
        res=rgb2gray(image);
    else
        res=image;
    end
    res=im2double(res);  %��ԭͼ��תΪ˫������������;
    [sel,ok]=listdlg('liststring',{'1','3','6','10','15','ȫ��'},...
        'listsize',[180 100],'OkString','ȷ��','CancelString','ȡ��',...
        'promptstring','ѡ��ѹ��DCTϵ��','name','ѹ��DCTϵ��ѡ��','selectionmode','single');
    if ok==1
        flag=1;
        T=dctmtx(8);  %������άDCT�任����
        B=blkproc(res,[8 8],'P1*x*P2',T,T');  %�����άDCT������T����ת��T'��DCT����P1*x*P2�Ĳ���
        if sel==1 %DCTϵ��Ϊ1
            tit='DCTϵ��Ϊ1';
            mask=[ 1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�1��
        elseif sel==2 %DCTϵ��Ϊ3
            tit='DCTϵ��Ϊ3';
            mask=[ 1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�3��
        elseif sel==3 %DCTϵ��Ϊ6
            tit='DCTϵ��Ϊ6';
            mask=[ 1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�6��
        elseif sel==4 %DCTϵ��Ϊ10
            tit='DCTϵ��Ϊ10';
            mask=[ 1 1 1 1 0 0 0 0
                1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�10��
        elseif sel==5 %DCTϵ��Ϊ15
            tit='DCTϵ��Ϊ15';
            mask=[ 1 1 1 1 1 0 0 0
                1 1 1 1 0 0 0 0
                1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�15��
        else  %ȫ��
            flag=0;
            mask1=[ 1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�1��
            
            mask3=[ 1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�3��
            
            mask6=[ 1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�6��
            
            mask10=[ 1 1 1 1 0 0 0 0
                1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�10��
            
            mask15=[ 1 1 1 1 1 0 0 0
                1 1 1 1 0 0 0 0
                1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %��ֵ��Ĥ��ѹ��DCTϵ����ֻ��DCTϵ�����Ͻ�15��
        end
        
        if flag==1
            B=blkproc(B,[8 8],' P1.*x',mask);   %����DCT�任��ϵ��
            I=blkproc(B,[8,8],'P1*x*P2',T',T); %��DCT���ع�ͼ��
            figure('toolbar','none','menubar','none','NumberTitle','off');
            imshow(I);
            title(tit);
        else
            B1=blkproc(B,[8 8],' P1.*x',mask1);   %����DCT�任��ϵ��
            I1= blkproc(B1,[8,8],'P1*x*P2',T',T); %��DCT���ع�ͼ��
            B3=blkproc(B,[8 8],' P1.*x',mask3);   %����DCT�任��ϵ��
            I3= blkproc(B3,[8,8],'P1*x*P2',T',T); %��DCT���ع�ͼ��
            B6=blkproc(B,[8 8],' P1.*x',mask6);   %����DCT�任��ϵ��
            I6= blkproc(B6,[8,8],'P1*x*P2',T',T); %��DCT���ع�ͼ��
            B10=blkproc(B,[8 8],' P1.*x',mask10);   %����DCT�任��ϵ��
            I10= blkproc(B10,[8,8],'P1*x*P2',T',T); %��DCT���ع�ͼ��
            B15=blkproc(B,[8 8],' P1.*x',mask15);   %����DCT�任��ϵ��
            I15= blkproc(B15,[8,8],'P1*x*P2',T',T); %��DCT���ع�ͼ��
            figure('toolbar','none','menubar','none','NumberTitle','off');
            subplot(2,3,1);
            imshow(I1);title('DCTϵ��Ϊ1');
            subplot(2,3,2);
            imshow(I3);title('DCTϵ��Ϊ3');
            subplot(2,3,3);
            imshow(I6);title('DCTϵ��Ϊ6');
            subplot(2,3,4);
            imshow(I10);title('DCTϵ��Ϊ10');
            subplot(2,3,5);
            imshow(I15);title('DCTϵ��Ϊ15');
        end
    end
else
    warndlg('��ѡ��ͼƬ','Waring');
end

