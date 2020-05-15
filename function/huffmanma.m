function  [CODE,s] =huffmanma(p)
global CODE
CODE=cell(length(p),1);
if length(p)>1
    p=p/sum(p);
    s=cell(length(p),1);
    for i=1:length(p)
        s{i}=i;
    end
    m=size(s);
    while length(s)>2
        [p,i]=sort(p);
        p(2)=p(1)+p(2);
        p(1)=[];
        s=s(i);
        s{2}={s{1},s{2}};
        s(1)=[];
    end
    makecode(s,[]);
else
    CODE={'1'};
end;
