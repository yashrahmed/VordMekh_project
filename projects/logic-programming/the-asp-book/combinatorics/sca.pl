% Sequence covering array SCA(n; 3, v)
% run with: clingo sca.pl -c v=4 -c n=6

% define the symbols, positions and rows
sym(1..v).   % the v symbols (events)
posn(1..v).  % the v positions in a row
row(1..n).   % the n rows (test runs)

% choice rule: place each symbol at exactly one position in each row
{ pos(R,S,P) : posn(P) } = 1 :- row(R), sym(S).

% constraint: no two symbols share a position in a row
:- row(R), posn(P), #count{ S : pos(R,S,P) } != 1.

% symmetry breaking: fix row 1 as the identity 1 2 ... v
pos(1,S,S) :- sym(S).

% cov(A,B,C): some row contains A ... B ... C in that order
cov(A,B,C) :- pos(R,A,PA), pos(R,B,PB), pos(R,C,PC), PA < PB, PB < PC.

% constraint: every sequence of 3 distinct symbols must be covered
:- sym(A), sym(B), sym(C), A != B, B != C, A != C, not cov(A,B,C).

#show pos/3.
