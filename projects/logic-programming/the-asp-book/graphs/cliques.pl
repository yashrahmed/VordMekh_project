
vertex(a; b; c; d).
edge(a,b; b,c; a,c; b,d).

% pick a single assignment for a vertex.
{ in(Grp, Ve) : Grp = (1..4) } = 1 :- vertex(Ve).

% define adjacency
adj(V1, V2) :- edge(V1, V2).
adj(V1, V2) :- edge(V2, V1).

% Rules for symmetry breaking. Sequence the assign
group_opened_before(G, V) :- in(G, U), vertex(V), U < V.        % some vertex before V is in G
:- in(G, V), G > 1, not group_opened_before(G-1, V).            % V may use G > 1 only if G-1 was used earlier

% define a clique as a constraint on every pair that is assigned to the same group.
:- in(S1, V1), in(S1, V2), V1 < V2, not adj(V1, V2).

#show in/2.

