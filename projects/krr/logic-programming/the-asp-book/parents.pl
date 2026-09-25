parent(ann,bob).
parent(bob,carol).
parent(bob,dan). % Facts are "rules" without a body.


child(X, Y) :- parent(Y, X).
