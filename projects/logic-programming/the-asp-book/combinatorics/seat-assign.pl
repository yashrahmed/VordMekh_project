%% DOMAIN

guest(1..N) :- size(N).


% Every guest G is allocated a single seat in range [1..N]; 1 choice per guest is required.
% The following results in a single choice for ALL guests-seat combo; Incorrect.
% { at(1..N, 1..N) } = 1 :- size(N).
{ at(G, 1..N) } = 1 :- guest(G), size(N).

% No 2 guests are allocated the same seat.
% G1 = G2 :- at(G1 , C), at(G2 , C). % This works but IMO is hard to read. Clingo will evaluate the
% correctness of G1=G2 independently which will eval to F and fail the rule and disqualify the model.
:- at(G1 , C), at(G2 , C), not G1 = G2.

% Define adjacency
adj(X, Y) :- at(_, X), at(_, Y), |X-Y|=1.
adj(N, 1) :- size(N).
adj(1, N) :- size(N).

% Place constraint on seating people who like each other next to each other.
:- at(A, X), at(B, Y), likes(A, B), not adj(X, Y).

% Place constraint on seating people who dislike each other at least one seat apart.
:- at(A, X), at(B, Y), dislikes(A, B), adj(X, Y).

% Not start with the problem specific facts.
size(6).
likes (1,2; 3,4).
dislikes (2,3; 1,3).

#show at/2.
