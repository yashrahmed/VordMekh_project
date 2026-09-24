% define the pattern as p(index, pattern_val)
p(1,1). p(2, 3). p(3, 2).

% define the text/input sequence as t(index, value)
t(1,3). t(2,1). t(3,4). t(4,2). t(5,5).

% define a map clause as m(pat_idx, text_idx). A stable model is made of a list of m(...,....)
{ m(I,J) : t(J,_) } = 1 :- p(I,_). % choice rule to map a pattern val to a text val.

% constraint to disqualify answer sets with non strictly increasing assignments.
:- m(I, J1), m(I+1, J2), J2 <= J1.

% pairwise constraint to ensure that the permutation pattern match.
% deliberately tests for strictly increasing p() pairs by value.
:- m(I1, J1), m(I2, J2), p(I1,P1), p(I2,P2),
 t(J1,T1), t(J2,T2), P1 < P2, T1 >= T2.

#show m/2.
