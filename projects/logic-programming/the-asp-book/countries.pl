#const c0=uk.

size(france,65; germany,83; italy,61; uk,64).

large(C) :- size(C,S1), size(c0,S2), S1 > S2.

#show large/1.
