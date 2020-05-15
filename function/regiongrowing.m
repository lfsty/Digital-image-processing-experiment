function J = regiongrowing(I,y,x,threshold)

if(exist('threshold','var')==0), threshold=0.2; end
J = zeros(size(I)); % 用来标记输出结果的二值矩阵
[m, n] = size(I); % 输入图像的尺寸
reg_mean = I(x,y); % 被分割区域的灰度均值
reg_size = 1; % 区域中像素的数目
% 用以存储被分割出来的区域的邻域点的堆栈
neg_free = 10000; neg_pos=0;
neg_list = zeros(neg_free,3);
delta=0; % 最新被引入的像素与区域灰度均值的差值

% 区域生长直至满足终止条件
while(delta<threshold && reg_size<numel(I))

    % 检测邻域像素，并判读是否将其划入区域
    for i = -1:1
        for j = -1:1
            xn = x + i; yn = y + j; % 计算邻域点的坐标
            % 检查邻域像素是否越界
            indicator = (xn >= 1)&&(yn >= 1)&&(xn <= m)&&(yn <= n);
        
            % 如果邻域像素还不属于被分割区域则加入堆栈
            if(indicator && (J(xn,yn)==0))
                neg_pos = neg_pos+1;
                neg_list(neg_pos,:) = [xn yn I(xn,yn)]; J(xn,yn)=1;
            end
        end
    end
    
    if(neg_pos+10>neg_free) % 如果堆栈空间不足，则对其进行扩容
        neg_free=neg_free+10000;
        neg_list((neg_pos+1):neg_free,:)=0;
    end
    
    % 将那些灰度值最接近区域均值的像素加入到区域中去
    dist = abs(neg_list(1:neg_pos,3)-reg_mean);
    [delta, index] = min(dist);
    J(x,y)=2; reg_size=reg_size+1;
    
    % 计算新区域的均值
    reg_mean = (reg_mean*reg_size + neg_list(index,3))/(reg_size+1);
    % 保存像素坐标，然后将像素从堆栈中移除
    x = neg_list(index,1); y = neg_list(index,2);
    neg_list(index,:)=neg_list(neg_pos,:); neg_pos=neg_pos-1;
end


