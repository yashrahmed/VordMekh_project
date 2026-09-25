% sq(N, N*N) :- N = 0..5. % '=' in this case implies 'is in set'
% p(X, Y) :- X = 1..4, Y = 1..X.

number(Tgt) :- Tgt=1..10.

divides(Tgt, N) :-  number(Tgt), N = 1..Tgt, Tgt\N = 0. % '\\' represents modulo and '\/' represents integer division

common_divisor(A, B, N) :- number(A), number(B), A > B, N=1..10, divides(A, N), divides(B, N).

common_div_above_one(A, B) :- number(A), number(B), common_divisor(A, B, X),  X > 1.

coprimes(A, B) :- number(A), number(B), B > 1,  A > B, not common_div_above_one(A, B).

#show coprimes/2.
