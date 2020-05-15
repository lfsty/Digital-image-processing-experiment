function makecode(sc,codeword)
global CODE
if isa(sc,'cell')
    makecode(sc{1},[codeword 0]);
    makecode(sc{2},[codeword 1]);
else
    CODE{sc}=char('0'+codeword);
end
