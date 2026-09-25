#const r = 5.
#const n = 20.

number(1..n).
color(1..r).

% Assign each number exactly one color.
1 { in(I,C) : color(C) } 1 :- number(I).

% Forbid monochromatic A+B=C.
% A <= B removes symmetric duplicates but retains A=B.
:- in(A,C), in(B,C), A <= B, A+B <= n, in(A+B,C).

#show in/2.
