function [A_tensor,tnn,trank] = solve_A_tensor(B_tensor,tau)
% Input:
%       B_tensor: tensor： n1*n2*n3 tensor 将高阶图组成的张量按照一定的模方向进行旋转得到
%       tau: the regular
% 
% Output: 
%       A_tensor: n1*n2*n3 tensor 
%
% The proximal operator of the tensor nuclear norm of a 3 way tensor
%
% min_A 0.5*||A-B||_F^2+rho*||A||_*
%
% Written by Rong Wang (wangrong07@tsinghua.org.cn), written in 2023/06/13

[n1,n2,n3] = size(B_tensor);
A_tensor  = zeros(n1,n2,n3);
B_tensor = fft(B_tensor,[],3);
tnn = 0;
trank = 0;
        
% first frontal slice
[U,S,V] = svd(B_tensor(:,:,1),'econ');
S = diag(S);
r = length(find(S>tau));
if r>=1
    S = S(1:r)-tau;
    A_tensor(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
    tnn = tnn+sum(S);
    trank = max(trank,r);
end
% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(B_tensor(:,:,i),'econ');
    S = diag(S);
    r = length(find(S>tau));
    if r>=1
        S = S(1:r)-tau;
        A_tensor(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        tnn = tnn+sum(S)*2;
        trank = max(trank,r);
    end
    A_tensor(:,:,n3+2-i) = conj(A_tensor(:,:,i));
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(B_tensor(:,:,i),'econ');
    S = diag(S);
    r = length(find(S>tau));
    if r>=1
        S = S(1:r)-tau;
        A_tensor(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        tnn = tnn+sum(S);
        trank = max(trank,r);
    end
end
tnn = tnn/n3;
A_tensor = ifft(A_tensor,[],3);

end
























