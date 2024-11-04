function Bho = high_order_bipartite_gen(B1,order)
% Input:
%        B: bipartite of size n*m  where n denotes the number of data
%           poits  and m denotes the number of anchors.
%
%        order: the order of bipartite graph
% 
% Output: 
%        Bho: the set of all bipartite graph
% Written by Rong Wang (wangrong07@tsinghua.org.cn), written in 2023/05/25

disp('----------Generate High Order Bipartite Graphs----------');

Bho = cell(order,1);     

sigma = sparse(diag(sum(B1).^(-0.5)));
Bho{1} = B1*sigma;

[U,S,V] = svd(B1,'econ');

for d = 2:order
    Temp = U*S.^(2*d-1)*V';
    Temp = Temp./max(max(Temp));
    Temp(Temp<1e-5) = 0; 
    Bho{d} = Temp;
end


