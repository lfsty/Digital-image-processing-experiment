function J = regiongrowing(I,y,x,threshold)

if(exist('threshold','var')==0), threshold=0.2; end
J = zeros(size(I)); % ��������������Ķ�ֵ����
[m, n] = size(I); % ����ͼ��ĳߴ�
reg_mean = I(x,y); % ���ָ�����ĻҶȾ�ֵ
reg_size = 1; % ���������ص���Ŀ
% ���Դ洢���ָ����������������Ķ�ջ
neg_free = 10000; neg_pos=0;
neg_list = zeros(neg_free,3);
delta=0; % ���±����������������ҶȾ�ֵ�Ĳ�ֵ

% ��������ֱ��������ֹ����
while(delta<threshold && reg_size<numel(I))

    % ����������أ����ж��Ƿ��仮������
    for i = -1:1
        for j = -1:1
            xn = x + i; yn = y + j; % ��������������
            % ������������Ƿ�Խ��
            indicator = (xn >= 1)&&(yn >= 1)&&(xn <= m)&&(yn <= n);
        
            % ����������ػ������ڱ��ָ�����������ջ
            if(indicator && (J(xn,yn)==0))
                neg_pos = neg_pos+1;
                neg_list(neg_pos,:) = [xn yn I(xn,yn)]; J(xn,yn)=1;
            end
        end
    end
    
    if(neg_pos+10>neg_free) % �����ջ�ռ䲻�㣬������������
        neg_free=neg_free+10000;
        neg_list((neg_pos+1):neg_free,:)=0;
    end
    
    % ����Щ�Ҷ�ֵ��ӽ������ֵ�����ؼ��뵽������ȥ
    dist = abs(neg_list(1:neg_pos,3)-reg_mean);
    [delta, index] = min(dist);
    J(x,y)=2; reg_size=reg_size+1;
    
    % ����������ľ�ֵ
    reg_mean = (reg_mean*reg_size + neg_list(index,3))/(reg_size+1);
    % �����������꣬Ȼ�����شӶ�ջ���Ƴ�
    x = neg_list(index,1); y = neg_list(index,2);
    neg_list(index,:)=neg_list(neg_pos,:); neg_pos=neg_pos-1;
end


