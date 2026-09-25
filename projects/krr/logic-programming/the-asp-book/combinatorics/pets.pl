
%2. The iguana is not owned by either Chuck or Duane.
%3. Neither the jackal nor the king cobra are owned by Mr. Foster.
%4. The llama does not belong to Duane.
%5. Abner, who does not own the king cobra, is not Mr. Gunter.
%6. Bruce and Mr. Foster are neighbors.
%7. Mr. Halevy is afraid of iguanas.

last_name(engels).
last_name(foster).
last_name(gunter).
last_name(halevy).

first_name(abner).
first_name(bruce).
first_name(duane).
first_name(chuck).

pet(llama).
pet(iguana).
pet(cobra).
pet(jackal).



% A person must have a single first and last name.
{ person(F,L) : first_name(F) } = 1 :- last_name(L).
{ person(F,L) : last_name(L) } = 1 :- first_name(F).

% A person can only own a single animal.
{ owns(Am, F, L) : pet(Am)} = 1 :- person(F, L).
{ owns(Am, F, L) : person(F, L)} = 1 :- pet(Am).

% Read constraints as disqualify if ....
:- owns(iguana, F, _), first_name(F), F = (chuck; duane).
:- owns(A, _, foster), A =(jackal; cobra).
:- owns(llama, duane, _).
:- owns(cobra, abner, _).
:- person(abner, gunter).
:- owns(iguana, _, halevy).

#show owns/3.
