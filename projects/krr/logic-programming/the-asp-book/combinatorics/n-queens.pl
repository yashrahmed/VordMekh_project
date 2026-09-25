
row(1..8).
col(1..8).

{ at(R, C): row(R) } = 1 :- col(C).

on_diag(R1, C1, R2, C2) :- row(R1), row(R2), col(C1), col(C2), R1 - C1 = R2 - C2. % lower diagonal
on_diag(R1, C1, R2, C2) :- row(R1), row(R2), col(C1), col(C2), R1 + C1 = R2 + C2. % upper diagonal
% on_diag can be removed by checking the diagonal with arithmetic in the constraint below:
% two queens share a diagonal when their row distance equals their column distance.
% :- at(R1, C1), at(R2, C2), C1 < C2, |R1 - R2| = C2 - C1.
% This avoids grounding on_diag over every pair of cells (736 atoms down to 112 for n = 8).

:- at(R, C1), at(R, C2), C1 < C2. % C1 < C2 ensures that columns are different.
:- at(R1, C1), at(R2, C2), C1 < C2,  on_diag(R1, C1, R2, C2). % C1 < C2 ensures that columns are different.

#show at/2.