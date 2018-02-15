danger([Id, 4], 1) :-
    print('caso1: diagnosis=4'), nl, !.

danger([Id, 2], 1) :-
    inHistory(Id, 'cancer'),
    print('caso2: diagnosis=2 and cancer in history'), nl, !.

danger([Id, 0], 1) :-
    inHistory(Id, 'cancer'),
    print('caso3: diagnosis=0 and cancer in hisotry'), nl, !.

danger([Id, 2], 2) :-
    inDescription(Id, 'smoker'),
    print('caso4: diagnosis=2 and smoker'), nl, !.

danger([Id, 0], 2) :-
    inDescription(Id, 'smoker'),
    print('caso5: diagnosis=0 and smoker'), nl !.

danger([Id, 2], 3) :-
    print('caso6: diagnosis=2 and not cancer in history and not smoker'), nl, !.