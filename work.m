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
%选取全局变量记录图像矩阵，文件名，文件路径
global image filename filepath;
%选取图片，记录文件名，文件路径
[filename,filepath]=uigetfile({'*.*'},'选择图片','./image');
%判断是否选取到图片
if filename~=0
    %显示文件名在窗口上
    set(handles.filename,'String',filename);
    %设定显示区域
    axes(handles.image);
    %根据文件名，文件路径读取图片
    image=imread([filepath,filename]);
    %显示图片
    imshow(image);
end


function popupmenu1_Callback(hObject, eventdata, handles)


function popupmenu1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function filename_CreateFcn(hObject, eventdata, handles)


function first_Callback(hObject, eventdata, handles)



%加
function add_Callback(hObject, eventdata, handles)

%全局变量，获取以读取的图片
global image
if isempty(image)~=1
    %选择需要加上的图片
    [filename1,filepath1]=uigetfile({'*.*'},'选择图片');
    if filename1~=0
        %读取需要加上的图片
        image1=imread([filepath1,filename1]);
        %创建一个新窗口，标题为‘加’，不显示toolbar，menubar与标号，创建一个一行两列的表格，在第一格显示原图片，标题为‘加上的图片’
        figure('name','加','toolbar','none','menubar','none','NumberTitle','off');subplot(1,2,1);imshow(image1);title('加上的图片');
        %获取原图片的大小
        [width,length,height]=size(image);
        %初始化矩阵
        res=zeros(width,length,height);
        %double类型转换成uint8
        res=uint8(res);
        %防止加上的图片与原图大小不一致，进行裁切
        image1=imresize(image1,[width length]);
        
        %两图相加
        for i=1:width
            for j=1:length
                for k=1:height
                    res(i,j,k)=image(i,j,k)+image1(i,j,k);
                end
            end
        end
        %在第二格显示图片，标题为‘加上的结果’
        subplot(1,2,2);imshow(res);title('加上后的结果');
    end
else
    warndlg('请选择图片','Waring');
end

%减
%与加大致相同，注释参考加
function div_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [filename1,filepath1]=uigetfile({'*.*'},'选择图片');
    if filename1~=0
        image1=imread([filepath1,filename1]);
        figure('name','减','toolbar','none','menubar','none','NumberTitle','off');subplot(1,2,1);imshow(image1);title('减取的图片');
        
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
        subplot(1,2,2);imshow(res);title('减去后的结果');
    end
else
    warndlg('请选择图片','Waring');
end



% 移动
function move_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    %创建一个输入对话框，获取移动的方向
    %以左上角为原点，向右为x轴正方向，向下为y轴正方向
    
    %提示字符串
    prompt={'x轴','y轴'};
    %对话框名称
    name='移动方向';
    %显示的行数
    numlines=1;
    %默认数值
    defaultanswer={'50','50'};
    
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    
    answer=inputdlg(prompt,name,numlines,defaultanswer,options);
    if isempty(answer)~=1
        %获取输入值，进行转化，获得移动方向与数值
        delY=answer(1);
        delX=answer(2);
        delX=cell2mat(delX);
        delY=cell2mat(delY);
        delX=str2num(delX);
        delY=str2num(delY);
        
        
        [width,length,height]=size(image);
        res=zeros(width,length,height);
        res=uint8(res);
        % 平移
        tras = [1 0 delX; 0 1 delY; 0 0 1]; % 平移的变换矩阵
        for i = 1 : width
            for j = 1 : length
                temp = [i; j; 1];
                temp = tras * temp; % 矩阵乘法
                x = temp(1, 1);
                y = temp(2, 1);
                % 变换后的位置判断是否越界
                if (x <= width) & (y <= length) & (x >= 1) & (y >= 1)
                    res(x, y,:) = image(i, j,:);
                end
            end
        end
        figure('name','移动','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('移动后的图片');
    end
else
    warndlg('请选择图片','Waring');
end

%镜像
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
    figure('name','镜像','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('镜像后的图片');
else
    warndlg('请选择图片','Waring');
end



% 旋转
function rotate_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    %提示字符串
    prompt={'旋转角度'};
    %对话框名称
    name='旋转';
    %显示的行数
    numlines=1;
    %默认数值
    defaultanswer={'30'};
    
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    
    answer=inputdlg(prompt,name,numlines,defaultanswer,options);
    if isempty(answer)~=1
        %获取输入值，进行转化，获得旋转角度
        ang=answer(1);
        ang=cell2mat(ang);
        ang=str2num(ang);
        
        res=imrotate(image,ang);
        figure('name','旋转','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('旋转后的图片');
    end
else
    warndlg('请选择图片','Waring');
end



function second_Callback(hObject, eventdata, handles)


%亮度变换
function bright_trans_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    res=image;
    %     image_size=size(image);
    %     dimension=numel(image_size);
    %     if dimension==3
    %         res=rgb2gray(image);
    %     end
    
    %提示字符串
    prompt={'low in  high in（两数之间用空格隔开）','low out high out（两数之间用空格隔开）','gamma'};
    %对话框名称
    name='亮度变换';
    %显示的行数
    numlines=1;
    %默认数值
    defaultanswer={'0.3 0.8','0 1','1'};
    
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    
    answer=inputdlg(prompt,name,numlines,defaultanswer,options);
    if isempty(answer)~=1
        %获取输入值，进行转化，获得low_in,high_in,low_out,high_out,gamma
        in=answer(1);
        out=answer(2);
        gam=answer(3);
        out=cell2mat(out);
        in=cell2mat(in);
        gam=cell2mat(gam);
        out=str2num(out);
        in=str2num(in);
        gam=str2num(gam);
        
        figure('name','直方图','toolbar','none','menubar','none','NumberTitle','off');
        subplot(1,2,1);imhist(res);title('原图的直方图');
        res=imadjust(res,in,out,gam);%进行亮度变换
        subplot(1,2,2);imhist(res);title('亮度变换后的直方图');
        figure('name','亮度变换','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('亮度变换后的图片');
    end
else
    warndlg('请选择图片','Waring');
end


% 直方图均衡
function Histogram_Equalization_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    res=histeq(image); %对原图像进行直方图均衡化处理
    figure('name','直方图均衡','toolbar','none','menubar','none','NumberTitle','off');imshow(res);title('直方图均衡后的图片');  %对原图像进行屏幕控制;显示直方图均衡化后的图像
    
    figure('name','直方图','toolbar','none','menubar','none','NumberTitle','off');
    %对直方图均衡化后的图像进行屏幕控制;作一幅子图作为并排两幅图的第1幅图,将原图像直方图显示为256级灰度,给原图像直方图加标题名
    subplot(1,2,1) ;imhist(image,256);  title('原图像直方图') ;
    %作第2幅子图,将均衡化后图像的直方图显示为256级灰度,给均衡化后图像直方图加标题名
    subplot(1,2,2);  imhist(res,256) ; title('均衡变换后的直方图') ;
else
    warndlg('请选择图片','Waring');
end


% laplacian算子空域滤波
function Spatial_filtering_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    
    w4=fspecial('laplacian',0);
    res=image;
    res=im2double(res);
    g4=res-imfilter(res,w4,'replicate');
    figure('name','laplacian算子空域滤波','toolbar','none','menubar','none','NumberTitle','off');imshow(g4);title('中心为-4拉普拉斯的效果');
    
else
    warndlg('请选择图片','Waring');
end


%二维傅里叶变换
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
    fftI=fft2(res);       %二维离散傅立叶变换
    sfftI=fftshift(fftI);  %直流分量移到频谱中心
    RR=real(sfftI);    %取傅立叶变换的实部
    II=imag(sfftI);     %取傅立叶变换的虚部
    A=sqrt(RR.^2+II.^2); %计算频谱幅值
    A=(A-min(min(A)))/(max(max(A))-min(min(A)))*225; %归一化
    figure('name','傅里叶变换','toolbar','none','menubar','none','NumberTitle','off');imshow(A);title('图像的频谱');
else
    warndlg('请选择图片','Waring');
end


% 巴特沃斯滤波器
function Butterworth_filters_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [sel,ok]=listdlg('liststring',{'低通','高通'},...
        'listsize',[180 80],'OkString','确定','CancelString','取消',...
        'promptstring','巴特沃斯滤波器选择','name','选择滤波器','selectionmode','single');
    if ok==1
        
        %提示字符串
        prompt={'请输入截止频率：'};
        %对话框名称
        name='截止频率';
        %显示的行数
        numlines=1;
        %默认数值
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
            J1=imnoise(res,'salt & pepper');                  %叠加校验噪声
            subplot(2,3,1);imshow(J1);title('叠加椒盐噪声图');
            f=double(J1);           %数据类型转换，MATLAB不支持图像的无符号整型的计算
            g=fft2(f);              %傅里叶变换
            g=fftshift(g);          %转换矩阵数据
            RR=real(g);%取傅里叶变换的实部
            II=imag(g);%取傅里叶变换的虚部
            A=sqrt(RR.^2+II.^2);%计算频谱幅值
            A=(A-min(min(A)))/(max(max(A))-min(min(A)))*225;%归一化
            subplot(2,3,2);imshow(A);title('滤波前的频谱');%显示原图像的频谱
            h1=g;[M,N]=size(g);nn=2;
            m=fix(M/2);n=fix(N/2);
            
            if sel==1%低通
                
                for i=1:M
                    for j=1:N
                        d=sqrt((i-m)^2+(j-n)^2);
                        h=1/(1+(d/d0)^(2*nn));%生成二阶巴特沃斯低通滤波器
                        h1(i,j)=h;
                    end
                end
                subplot(2,3,3);imshow(h1);title('低通滤波器的图像');%显示滤波器的图像
                
            elseif sel==2%高通
                
                for i=1:M
                    for j=1:N
                        d=sqrt((i-m)^2+(j-n)^2);
                        h=1/(1+(d0/d)^(2*nn));   %计算高通滤波器传递函数
                        h2=0.5+2*h;    %设计high-frequency emphasis其中a=0.5,b=2.0
                        h1(i,j)=h2;  %用设计的滤波器处理原图像
                    end
                end
                subplot(2,3,3);imshow(h1);title('高通滤波器的图像');%显示滤波器的图像
                
            end
            
            result=h1.*g;          %滤波处理
            RR1=real(result);      %取傅里叶变换的实部
            II1=imag(result);      %取傅里叶变换的虚部
            A1=sqrt(RR1.^2+II1.^2);%计算频谱幅值
            A1=(A1-min(min(A1)))/(max(max(A1))-min(min(A1)))*225;%归一化
            subplot(2,3,4);imshow(A1);title('滤波后的频谱'); %显示滤波后的频谱
            result=ifftshift(result);%滤波处理
            J2=ifft2(result);%傅里叶反变换
            J3=uint8(real(J2));%取实部
            subplot(2,3,5);imshow(J3);title('滤波后的图像');%显示滤波效果
        end
    end
else
    warndlg('请选择图片','Waring');
end

function thrid_Callback(hObject, eventdata, handles)

% 中值滤波
function mid_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [sel,ok]=listdlg('liststring',{'椒盐噪声','高斯噪声','泊松噪声','乘法噪声'},...
        'listsize',[180 80],'OkString','确定','CancelString','取消',...
        'promptstring','选择叠加的噪声','name','噪声选择','selectionmode','single');
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
        
        if sel==1       %椒盐噪声
            %提示字符串
            prompt={'请输入椒盐噪声的密度：'};
            %对话框名称
            name='椒盐噪声';
            %显示的行数
            numlines=1;
            %默认数值
            defaultanswer={'0.05'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                m=answer(1);
                m=cell2mat(m);
                m=str2double(m);
                J2=imnoise(res,'salt & pepper',m);
                
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('椒盐噪声');
                I_Filter1=medfilt2(J2,[3 3]);%窗口大小为3*3
                subplot(1,4,2);imshow(I_Filter1);title('3*3中值滤波');
                I_Filter2=medfilt2(J2,[5 5]);%窗口大小为5*5
                subplot(1,4,3);imshow(I_Filter2);title('5*5中值滤波');
                I_Filter3=medfilt2(J2,[7 7]);%窗口大小为7*7
                subplot(1,4,4);imshow(I_Filter3);title('7*7中值滤波');
            end
        elseif sel==2   %高斯噪声
            %提示字符串
            prompt={'请输入高斯噪声的均值：','请输入高斯噪声的方差'};
            %对话框名称
            name='高斯噪声';
            %显示的行数
            numlines=1;
            %默认数值
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
                subplot(1,4,1);imshow(J2);title('高斯噪声');
                I_Filter1=medfilt2(J2,[3 3]);%窗口大小为3*3
                subplot(1,4,2);imshow(I_Filter1);title('3*3中值滤波');
                I_Filter2=medfilt2(J2,[5 5]);%窗口大小为5*5
                subplot(1,4,3);imshow(I_Filter2);title('5*5中值滤波');
                I_Filter3=medfilt2(J2,[7 7]);%窗口大小为7*7
                subplot(1,4,4);imshow(I_Filter3);title('7*7中值滤波');
            end
        elseif sel==3   %泊松噪声
            
            J2=imnoise(res,'poisson');
            figure('toolbar','none','menubar','none','NumberTitle','off');
            subplot(1,4,1);imshow(J2);title('泊松噪声');
            I_Filter1=medfilt2(J2,[3 3]);%窗口大小为3*3
            subplot(1,4,2);imshow(I_Filter1);title('3*3中值滤波');
            I_Filter2=medfilt2(J2,[5 5]);%窗口大小为5*5
            subplot(1,4,3);imshow(I_Filter2);title('5*5中值滤波');
            I_Filter3=medfilt2(J2,[7 7]);%窗口大小为7*7
            subplot(1,4,4);imshow(I_Filter3);title('7*7中值滤波');
            
        elseif sel==4   %乘法噪声
            %提示字符串
            prompt={'请输入乘法噪声的方差（均值为0）：'};
            %对话框名称
            name='乘法噪声';
            %显示的行数
            numlines=1;
            %默认数值
            defaultanswer={'0.04'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                
                v=answer(1);
                v=cell2mat(v);
                v=str2double(v);
                J2=imnoise(res,'speckle',v);
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('乘法噪声');
                I_Filter1=medfilt2(J2,[3 3]);%窗口大小为3*3
                subplot(1,4,2);imshow(I_Filter1);title('3*3中值滤波');
                I_Filter2=medfilt2(J2,[5 5]);%窗口大小为5*5
                subplot(1,4,3);imshow(I_Filter2);title('5*5中值滤波');
                I_Filter3=medfilt2(J2,[7 7]);%窗口大小为7*7
                subplot(1,4,4);imshow(I_Filter3);title('7*7中值滤波');
            end
        end
    end
else
    warndlg('请选择图片','Waring');
end

% 均值滤波
function average_Callback(hObject, eventdata, handles)
global image
if isempty(image)~=1
    [sel,ok]=listdlg('liststring',{'椒盐噪声','高斯噪声','泊松噪声','乘法噪声'},...
        'listsize',[180 80],'OkString','确定','CancelString','取消',...
        'promptstring','选择叠加的噪声','name','噪声选择','selectionmode','single');
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
        
        if sel==1       %椒盐噪声
            %提示字符串
            prompt={'请输入椒盐噪声的密度：'};
            %对话框名称
            name='椒盐噪声';
            %显示的行数
            numlines=1;
            %默认数值
            defaultanswer={'0.05'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                m=answer(1);
                m=cell2mat(m);
                m=str2double(m);
                J2=imnoise(res,'salt & pepper',m);
                
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('椒盐噪声');
                I_Filter1=filter2(fspecial('average',3),J2)/255;%进行3*3的均值滤波
                subplot(1,4,2);imshow(I_Filter1);title('3*3模板均值滤波');
                I_Filter2=filter2(fspecial('average',5),J2)/255;%进行5*5的均值滤波
                subplot(1,4,3);imshow(I_Filter2);title('5*5模板均值滤波');
                I_Filter3=filter2(fspecial('average',7),J2)/255;%进行7*7的均值滤波
                subplot(1,4,4);imshow(I_Filter3);title('7*7模板均值滤波');
            end
        elseif sel==2   %高斯噪声
            %提示字符串
            prompt={'请输入高斯噪声的均值：','请输入高斯噪声的方差'};
            %对话框名称
            name='高斯噪声';
            %显示的行数
            numlines=1;
            %默认数值
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
                subplot(1,4,1);imshow(J2);title('高斯噪声');
                I_Filter1=filter2(fspecial('average',3),J2)/255;%进行3*3的均值滤波
                subplot(1,4,2);imshow(I_Filter1);title('3*3模板均值滤波');
                I_Filter2=filter2(fspecial('average',5),J2)/255;%进行5*5的均值滤波
                subplot(1,4,3);imshow(I_Filter2);title('5*5模板均值滤波');
                I_Filter3=filter2(fspecial('average',7),J2)/255;%进行7*7的均值滤波
                subplot(1,4,4);imshow(I_Filter3);title('7*7模板均值滤波');
            end
        elseif sel==3   %泊松噪声
            
            J2=imnoise(res,'poisson');
            figure('toolbar','none','menubar','none','NumberTitle','off');
            subplot(1,4,1);imshow(J2);title('泊松噪声');
            I_Filter1=filter2(fspecial('average',3),J2)/255;%进行3*3的均值滤波
            subplot(1,4,2);imshow(I_Filter1);title('3*3模板均值滤波');
            I_Filter2=filter2(fspecial('average',5),J2)/255;%进行5*5的均值滤波
            subplot(1,4,3);imshow(I_Filter2);title('5*5模板均值滤波');
            I_Filter3=filter2(fspecial('average',7),J2)/255;%进行7*7的均值滤波
            subplot(1,4,4);imshow(I_Filter3);title('7*7模板均值滤波');
            
        elseif sel==4   %乘法噪声
            %提示字符串
            prompt={'请输入乘法噪声的方差（均值为0）：'};
            %对话框名称
            name='乘法噪声';
            %显示的行数
            numlines=1;
            %默认数值
            defaultanswer={'0.04'};
            
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            if isempty(answer)~=1
                
                v=answer(1);
                v=cell2mat(v);
                v=str2double(v);
                J2=imnoise(res,'speckle',v);
                figure('toolbar','none','menubar','none','NumberTitle','off');
                subplot(1,4,1);imshow(J2);title('乘法噪声');
                I_Filter1=filter2(fspecial('average',3),J2)/255;%进行3*3的均值滤波
                subplot(1,4,2);imshow(I_Filter1);title('3*3模板均值滤波');
                I_Filter2=filter2(fspecial('average',5),J2)/255;%进行5*5的均值滤波
                subplot(1,4,3);imshow(I_Filter2);title('5*5模板均值滤波');
                I_Filter3=filter2(fspecial('average',7),J2)/255;%进行7*7的均值滤波
                subplot(1,4,4);imshow(I_Filter3);title('7*7模板均值滤波');
            end
        end
    end
else
    warndlg('请选择图片','Waring');
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
    
    %点检测
    w = [-1 -1 -1;-1 8 -1;-1 -1 -1];                    % 点检测掩模
    g = abs(imfilter(double(res),w));
    T = max(g(:));
    g = g>=T;
    figure('toolbar','none','menubar','none','NumberTitle','off');imshow(g);title('点检测');
    
else
    warndlg('请选择图片','Waring');
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
    
    [sel,ok]=listdlg('liststring',{'水平模板','45°模板','垂直模板','-45°模板'},...
        'listsize',[180 80],'OkString','确定','CancelString','取消',...
        'promptstring','选择模板','name','模板选择','selectionmode','single');
    if ok==1
        %线检测
        if sel==1
            w = [-1,-1,-1;2,2,2;-1,-1,-1];%水平模板
            tit='水平模板';
        elseif sel==2
            w = [-1,-1,2;-1,2,-1;2,-1,-1];%45°模板
            tit='45°模板';
        elseif sel==3
            w = [-1,2,-1;-1,2,-1;-1,2,-1];%垂直模板
            tit='垂直模板';
        elseif sel==4
            w = [2 -1 -1;-1 2 -1;-1 -1 2];%-45°模板
            tit='-45°模板';
        end
        
        g = imfilter(double(res),w);
        g1 = abs(g);                             % 检测图的绝对值
        T = max(g1(:));
        g2 = g1>=T;
        
        figure('toolbar','none','menubar','none','NumberTitle','off');
        subplot(1,2,1);imshow(g1,[]);title(tit);
        subplot(1,2,2);imshow(g2);title('g>=T');
    end
else
    warndlg('请选择图片','Waring');
end

%边缘检测
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
    
    answer=questdlg('是否添加噪声？','添加噪声','Yes','No','Yes');
    if strcmp(answer,'Yes')
        res=imnoise(res,'gaussian',0,0.01);
    end
    
    BW1=edge(res,'sobel');
    BW2=edge(res,'roberts');
    BW3=edge(res,'log');
    BW4=edge(res,'canny');
    BW5=edge(res,'prewitt');
    
    figure('toolbar','none','menubar','none','NumberTitle','off');
    subplot(2,3,1);imshow(res);title('原图像');
    subplot(2,3,2);imshow(BW1);title('sobel检测结果');
    subplot(2,3,3);imshow(BW2);title('roberts检测结果');
    subplot(2,3,4);imshow(BW3);title('log检测结果');
    subplot(2,3,5);imshow(BW4);title('canny检测结果');
    subplot(2,3,6);imshow(BW5);title('prewitt检测结果');
    
    
else
    warndlg('请选择图片','Waring');
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
    
    msgbox('请选取种子位置');
    uiwait;
    
    %区域生长法
    [x,y] = ginput(1);
    x=fix(x);
    y=fix(y);
    while x>length || y>width || x<0 || y<0
        msgbox('请在图片中选取种子位置');
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
    warndlg('请选择图片','Waring');
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
    %切割图片
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
    
    %提示字符串
    prompt={'请输入相似度：'};
    %对话框名称
    name='相似度';
    %显示的行数
    numlines=1;
    %默认数值
    defaultanswer={'0.27'};
    
    options.Resize='on';
    options.WindowStyle='normal';
    options.Interpreter='tex';
    
    answer=inputdlg(prompt,name,numlines,defaultanswer,options);
    if isempty(answer)~=1
        
        v=answer(1);
        v=cell2mat(v);
        v=str2double(v);
        
        S = qtdecomp(res,v);%相似度准则v
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
    warndlg('请选择图片','Waring');
end


function fifth_Callback(hObject, eventdata, handles)

%DCT频谱
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
    warndlg('请选择图片','Waring');
end

%图像压缩
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
    res=im2double(res);  %将原图像转为双精度数据类型;
    [sel,ok]=listdlg('liststring',{'1','3','6','10','15','全部'},...
        'listsize',[180 100],'OkString','确定','CancelString','取消',...
        'promptstring','选择压缩DCT系数','name','压缩DCT系数选择','selectionmode','single');
    if ok==1
        flag=1;
        T=dctmtx(8);  %产生二维DCT变换矩阵
        B=blkproc(res,[8 8],'P1*x*P2',T,T');  %计算二维DCT，矩阵T及其转置T'是DCT函数P1*x*P2的参数
        if sel==1 %DCT系数为1
            tit='DCT系数为1';
            mask=[ 1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角1个
        elseif sel==2 %DCT系数为3
            tit='DCT系数为3';
            mask=[ 1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角3个
        elseif sel==3 %DCT系数为6
            tit='DCT系数为6';
            mask=[ 1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角6个
        elseif sel==4 %DCT系数为10
            tit='DCT系数为10';
            mask=[ 1 1 1 1 0 0 0 0
                1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角10个
        elseif sel==5 %DCT系数为15
            tit='DCT系数为15';
            mask=[ 1 1 1 1 1 0 0 0
                1 1 1 1 0 0 0 0
                1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角15个
        else  %全部
            flag=0;
            mask1=[ 1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角1个
            
            mask3=[ 1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角3个
            
            mask6=[ 1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角6个
            
            mask10=[ 1 1 1 1 0 0 0 0
                1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角10个
            
            mask15=[ 1 1 1 1 1 0 0 0
                1 1 1 1 0 0 0 0
                1 1 1 0 0 0 0 0
                1 1 0 0 0 0 0 0
                1 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0]; %二值掩膜，压缩DCT系数，只留DCT系数左上角15个
        end
        
        if flag==1
            B=blkproc(B,[8 8],' P1.*x',mask);   %保留DCT变换的系数
            I=blkproc(B,[8,8],'P1*x*P2',T',T); %逆DCT，重构图像
            figure('toolbar','none','menubar','none','NumberTitle','off');
            imshow(I);
            title(tit);
        else
            B1=blkproc(B,[8 8],' P1.*x',mask1);   %保留DCT变换的系数
            I1= blkproc(B1,[8,8],'P1*x*P2',T',T); %逆DCT，重构图像
            B3=blkproc(B,[8 8],' P1.*x',mask3);   %保留DCT变换的系数
            I3= blkproc(B3,[8,8],'P1*x*P2',T',T); %逆DCT，重构图像
            B6=blkproc(B,[8 8],' P1.*x',mask6);   %保留DCT变换的系数
            I6= blkproc(B6,[8,8],'P1*x*P2',T',T); %逆DCT，重构图像
            B10=blkproc(B,[8 8],' P1.*x',mask10);   %保留DCT变换的系数
            I10= blkproc(B10,[8,8],'P1*x*P2',T',T); %逆DCT，重构图像
            B15=blkproc(B,[8 8],' P1.*x',mask15);   %保留DCT变换的系数
            I15= blkproc(B15,[8,8],'P1*x*P2',T',T); %逆DCT，重构图像
            figure('toolbar','none','menubar','none','NumberTitle','off');
            subplot(2,3,1);
            imshow(I1);title('DCT系数为1');
            subplot(2,3,2);
            imshow(I3);title('DCT系数为3');
            subplot(2,3,3);
            imshow(I6);title('DCT系数为6');
            subplot(2,3,4);
            imshow(I10);title('DCT系数为10');
            subplot(2,3,5);
            imshow(I15);title('DCT系数为15');
        end
    end
else
    warndlg('请选择图片','Waring');
end

