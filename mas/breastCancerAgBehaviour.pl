danger([Id, 4], 1) :-
    print('case1: diagnosis=4'), nl, !.

danger([Id, 2], 1) :-
    inHistory(Id, 'cancer'),
    print('case2: diagnosis=2 and cancer in history'), nl, !.

danger([Id, 0], 1) :-
    inHistory(Id, 'cancer'),
    print('case3: diagnosis=0 and cancer in hisotry'), nl, !.

danger([Id, 2], 2) :-
    inDescription(Id, 'smoker'),
    print('case4: diagnosis=2 and smoker'), nl, !.

danger([Id, 0], 2) :-
    inDescription(Id, 'smoker'),
    print('case5: diagnosis=0 and smoker'), nl !.

danger([Id, 2], 3) :-
    print('case6: diagnosis=2 and not cancer in history and not smoker'), nl, !.

danger([Id, 0], 3) :-
    print('case7: diagnosis=0 and not cancer in history and not smoker'), nl, !.
