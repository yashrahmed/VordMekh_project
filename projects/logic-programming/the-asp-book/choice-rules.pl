%2 {p(1..3)}. % This reads a select a subset from {p(1), p(2), p(3), p(4)} which has 2 more more elements.
% The above can be written as 2 {p(1); p(2); p(3)}

% {p(X); q(X)} = K is equivalent to K {p(X); q{X}} K

person(ann; bob; carol; dan; elaine; fred).

eligible(ann; dan; fred).

% #1 — Nondeterministically choose exactly two eligible people.
{ elected(X) : person(X), eligible(X) } = 2.

% #2 — For each eligible person, require a singleton choice to contain
% exactly two atoms. This is unsatisfiable if any eligible person exists.
% { elected(X) } = 2 :- person(X), eligible(X).

% Basically, the grounding must populate the entire set elements inline for a choice to be made. THis
% does not happen in the case #2. Case #2 leads to 5 separate singleton choice rules.

% In Clingo, variables are never shared between rules.
% A variable can be "Global" only in the context of its own rule.
% A variable is "Local" if it appears ONLY between {} i.e. in aggregations or conditions aka the atoms after ':' in a choice rule.
% ANY variable not "Local" is "Global".

% The following rule is a constraint i.e. a rule without head. Read this as False :- conditions.
% Adding the following eliminates, models where dan gets elected.
:- elected(dan).

#show elected/1.
