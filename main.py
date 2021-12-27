import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from read_data import get_leagues_data

# https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html

# Variables for universes
unv_clubs = unv_goals = unv_avgValue = unv_players = unv_teamsChL = unv_teamsEuL = unv_teamsCfL = unv_score = None

# Variables for fuzzy membership functions
clubs_few = clubs_some = clubs_several = clubs_many = None
goals_few = goals_moderate = goals_many = None
avgValue_vpoor = avgValue_poor = avgValue_mpoor = avgValue_moderate = avgValue_mrich = avgValue_rich = avgValue_vrich = None
players_none = players_few = players_moderate = players_many = None
teamsChL_min = teamsChL_mid = teamsChL_max = None
teamsEuL_min = teamsEuL_mid = teamsEuL_max = None
teamsCfL_min = teamsCfL_mid = teamsCfL_max = None
score_tragic = score_bad = score_moderate = score_good = score_best = None

# Activations of fuzzy membership functions
clubs_few_level = clubs_some_level = clubs_several_level = clubs_many_level = None
goals_few_level = goals_moderate_level = goals_many_level = None
avgValue_vpoor_level = avgValue_poor_level = avgValue_mpoor_level = avgValue_moderate_level = avgValue_mrich_level = avgValue_rich_level = avgValue_vrich_level = None
players_none_level = players_few_level = players_moderate_level = players_many_level = None
teamsChL_min_level = teamsChL_mid_level = teamsChL_max_level = None
teamsEuL_min_level = teamsEuL_mid_level = teamsEuL_max_level = None
teamsCfL_min_level = teamsCfL_mid_level = teamsCfL_max_level = None


def initFuzzyMembershipFunctions():
    global unv_clubs, unv_goals, unv_avgValue, unv_players, unv_teamsChL, unv_teamsEuL, unv_teamsCfL, unv_score
    global clubs_few, clubs_some, clubs_several, clubs_many
    global goals_few, goals_moderate, goals_many
    global avgValue_vpoor, avgValue_poor, avgValue_mpoor, avgValue_moderate, avgValue_mrich, avgValue_rich, avgValue_vrich
    global players_none, players_few, players_moderate, players_many
    global teamsChL_min, teamsChL_mid, teamsChL_max
    global teamsEuL_min, teamsEuL_mid, teamsEuL_max
    global teamsCfL_min, teamsCfL_mid, teamsCfL_max
    global score_tragic, score_bad, score_moderate, score_good, score_best

    # Universes for variables
    unv_clubs = np.arange(2, 31, 1)
    unv_goals = np.arange(0.99, 5.01, 0.01)
    unv_avgValue = np.arange(0.01, 500.01, 0.01)
    unv_players = np.arange(0, 101, 1)
    unv_teamsChL = np.arange(0, 6, 1)
    unv_teamsEuL = np.arange(0, 6, 1)
    unv_teamsCfL = np.arange(0, 6, 1)
    # Result score universum
    unv_score = np.arange(0, 101, 1)

    # Generate fuzzy membership functions
    # trimf - triangular function
    # trapmf - trapezoidal function
    clubs_few = fuzz.trapmf(unv_clubs, [2, 2, 5, 8])
    clubs_some = fuzz.trapmf(unv_clubs, [3, 6, 9, 12])
    clubs_several = fuzz.trapmf(unv_clubs, [8, 11, 15, 18])
    clubs_many = fuzz.trapmf(unv_clubs, [15, 18, 30, 30])

    goals_few = fuzz.trapmf(unv_goals, [1.0, 1.0, 2.0, 3.0])
    goals_moderate = fuzz.trimf(unv_goals, [2.5, 3.0, 3.5])
    goals_many = fuzz.trapmf(unv_goals, [3.0, 3.5, 5.0, 5.0])
    
    avgValue_vpoor = fuzz.trapmf(unv_avgValue, [0.01, 0.01, 1.0, 2.0])
    avgValue_poor = fuzz.trimf(unv_avgValue, [1.0, 5.0, 15.0])
    avgValue_mpoor = fuzz.trimf(unv_avgValue, [7.5, 20.0, 50.0])
    avgValue_moderate = fuzz.trimf(unv_avgValue, [20.0, 50.0, 100.0])
    avgValue_mrich = fuzz.trimf(unv_avgValue, [50.0, 100.0, 175.0])
    avgValue_rich = fuzz.trimf(unv_avgValue, [150.0, 200.0, 275.0])
    avgValue_vrich = fuzz.trapmf(unv_avgValue, [200.0, 300.0, 500.0, 500.0])

    players_none = fuzz.trimf(unv_players, [0, 0, 1])
    players_few = fuzz.trapmf(unv_players, [1, 5, 10, 15])
    players_moderate = fuzz.trapmf(unv_players, [10, 15, 20, 25])
    players_many = fuzz.trapmf(unv_players, [20, 25, 100, 100])

    teamsChL_min = fuzz.trapmf(unv_teamsChL, [0, 0, 1, 2])
    teamsChL_mid = fuzz.trapmf(unv_teamsChL, [1, 2, 3, 4])
    teamsChL_max = fuzz.trapmf(unv_teamsChL, [3, 4, 5, 5])

    teamsEuL_min = fuzz.trimf(unv_teamsEuL, [0, 0, 1])
    teamsEuL_mid = fuzz.trimf(unv_teamsEuL, [0, 1, 2])
    teamsEuL_max = fuzz.trapmf(unv_teamsEuL, [1, 2, 5, 5])

    teamsCfL_min = fuzz.trapmf(unv_teamsCfL, [0, 0, 1, 2])
    teamsCfL_mid = fuzz.trimf(unv_teamsCfL, [1, 2, 3])
    teamsCfL_max = fuzz.trapmf(unv_teamsCfL, [2, 3, 5, 5])

    score_tragic = fuzz.trapmf(unv_score, [0, 0, 15, 25])
    score_bad = fuzz.trapmf(unv_score, [15, 25, 35, 45])
    score_moderate = fuzz.trapmf(unv_score, [35, 45, 55, 65])
    score_good = fuzz.trapmf(unv_score, [55, 65, 75, 85])
    score_best = fuzz.trapmf(unv_score, [75, 85, 100, 100])


def activateFuzzyMembershipFunctions(values):
    global unv_clubs, unv_goals, unv_avgValue, unv_players, unv_teamsChL, unv_teamsEuL, unv_teamsCfL
    global clubs_few, clubs_some, clubs_several, clubs_many
    global goals_few, goals_moderate, goals_many
    global avgValue_vpoor, avgValue_poor, avgValue_mpoor, avgValue_moderate, avgValue_mrich, avgValue_rich, avgValue_vrich
    global players_none, players_few, players_moderate, players_many
    global teamsChL_min, teamsChL_mid, teamsChL_max
    global teamsEuL_min, teamsEuL_mid, teamsEuL_max
    global teamsCfL_min, teamsCfL_mid, teamsCfL_max
    global clubs_few_level, clubs_some_level, clubs_several_level, clubs_many_level
    global goals_few_level, goals_moderate_level, goals_many_level
    global avgValue_vpoor_level, avgValue_poor_level, avgValue_mpoor_level, avgValue_moderate_level, avgValue_mrich_level, avgValue_rich_level, avgValue_vrich_level
    global players_none_level, players_few_level, players_moderate_level, players_many_level
    global teamsChL_min_level, teamsChL_mid_level, teamsChL_max_level
    global teamsEuL_min_level, teamsEuL_mid_level, teamsEuL_max_level
    global teamsCfL_min_level, teamsCfL_mid_level, teamsCfL_max_level

    # Activation of fuzzy membership functions at given values.
    clubs_few_level = fuzz.interp_membership(unv_clubs, clubs_few, values['clubs'])
    clubs_some_level = fuzz.interp_membership(unv_clubs, clubs_some, values['clubs'])
    clubs_several_level = fuzz.interp_membership(unv_clubs, clubs_several, values['clubs'])
    clubs_many_level = fuzz.interp_membership(unv_clubs, clubs_many, values['clubs'])

    goals_few_level = fuzz.interp_membership(unv_goals, goals_few, values['goalsPerMatch'])
    goals_moderate_level = fuzz.interp_membership(unv_goals, goals_moderate, values['goalsPerMatch'])
    goals_many_level = fuzz.interp_membership(unv_goals, goals_many, values['goalsPerMatch'])
    
    avgValue_vpoor_level = fuzz.interp_membership(unv_avgValue, avgValue_vpoor, values['avgMarketValueInMln'])
    avgValue_poor_level = fuzz.interp_membership(unv_avgValue, avgValue_poor, values['avgMarketValueInMln'])
    avgValue_mpoor_level = fuzz.interp_membership(unv_avgValue, avgValue_mpoor, values['avgMarketValueInMln'])
    avgValue_moderate_level = fuzz.interp_membership(unv_avgValue, avgValue_moderate, values['avgMarketValueInMln'])
    avgValue_mrich_level = fuzz.interp_membership(unv_avgValue, avgValue_mrich, values['avgMarketValueInMln'])
    avgValue_rich_level = fuzz.interp_membership(unv_avgValue, avgValue_rich, values['avgMarketValueInMln'])
    avgValue_vrich_level = fuzz.interp_membership(unv_avgValue, avgValue_vrich, values['avgMarketValueInMln'])

    players_none_level = fuzz.interp_membership(unv_players, players_none, values['famousPlayers'])
    players_few_level = fuzz.interp_membership(unv_players, players_few, values['famousPlayers'])
    players_moderate_level = fuzz.interp_membership(unv_players, players_moderate, values['famousPlayers'])
    players_many_level = fuzz.interp_membership(unv_players, players_many, values['famousPlayers'])

    teamsChL_min_level = fuzz.interp_membership(unv_teamsChL, teamsChL_min, values['teamsInChampionsLeague'])
    teamsChL_mid_level = fuzz.interp_membership(unv_teamsChL, teamsChL_mid, values['teamsInChampionsLeague'])
    teamsChL_max_level = fuzz.interp_membership(unv_teamsChL, teamsChL_max, values['teamsInChampionsLeague'])

    teamsEuL_min_level = fuzz.interp_membership(unv_teamsEuL, teamsEuL_min, values['teamsInEuropaLeague'])
    teamsEuL_mid_level = fuzz.interp_membership(unv_teamsEuL, teamsEuL_mid, values['teamsInEuropaLeague'])
    teamsEuL_max_level = fuzz.interp_membership(unv_teamsEuL, teamsEuL_max, values['teamsInEuropaLeague'])

    teamsCfL_min_level = fuzz.interp_membership(unv_teamsCfL, teamsCfL_min, values['teamsInConferenceLeague'])
    teamsCfL_mid_level = fuzz.interp_membership(unv_teamsCfL, teamsCfL_mid, values['teamsInConferenceLeague'])
    teamsCfL_max_level = fuzz.interp_membership(unv_teamsCfL, teamsCfL_max, values['teamsInConferenceLeague'])


def defineRules():
    global clubs_few_level, clubs_some_level, clubs_several_level, clubs_many_level
    global goals_few_level, goals_moderate_level, goals_many_level
    global avgValue_vpoor_level, avgValue_poor_level, avgValue_mpoor_level, avgValue_moderate_level, avgValue_mrich_level, avgValue_rich_level, avgValue_vrich_level
    global players_none_level, players_few_level, players_moderate_level, players_many_level
    global teamsChL_min_level, teamsChL_mid_level, teamsChL_max_level
    global teamsEuL_min_level, teamsEuL_mid_level, teamsEuL_max_level
    global teamsCfL_min_level, teamsCfL_mid_level, teamsCfL_max_level
    global score_tragic, score_bad, score_moderate, score_good, score_best, unv_score

    # OR
    active_rule1 = np.fmax(clubs_some_level,
                    np.fmax(clubs_few_level,
                        np.fmax(goals_few_level,
                            np.fmax(avgValue_mpoor_level,
                                np.fmax(avgValue_poor_level,
                                    np.fmax(avgValue_vpoor_level,
                                        np.fmax(players_none_level,
                                            np.fmax(teamsChL_min_level,
                                                np.fmax(teamsEuL_min_level, 
                                                    np.fmax(teamsCfL_max_level, teamsCfL_mid_level))))))))))

    active_rule2 = np.fmax(clubs_several_level,
                    np.fmax(clubs_some_level,
                        np.fmax(clubs_few_level,
                            np.fmax(goals_moderate_level,
                                np.fmax(goals_few_level,
                                    np.fmax(avgValue_moderate_level,
                                        np.fmax(avgValue_mpoor_level,
                                            np.fmax(avgValue_poor_level,
                                                np.fmax(players_few_level,
                                                    np.fmax(players_none_level,
                                                        np.fmax(teamsChL_min_level,
                                                            np.fmax(teamsEuL_min_level, 
                                                                np.fmax(teamsCfL_max_level, teamsCfL_mid_level)))))))))))))

    active_rule3 = np.fmax(clubs_several_level,
                    np.fmax(clubs_some_level,
                        np.fmax(goals_moderate_level,
                            np.fmax(avgValue_mrich_level,
                                np.fmax(avgValue_moderate_level,
                                    np.fmax(avgValue_mpoor_level,
                                        np.fmax(players_moderate_level,
                                            np.fmax(players_few_level,
                                                np.fmax(teamsChL_mid_level,
                                                    np.fmax(teamsChL_min_level,
                                                        np.fmax(teamsEuL_mid_level, 
                                                            np.fmax(teamsEuL_min_level, 
                                                                np.fmax(teamsCfL_max_level, teamsCfL_mid_level)))))))))))))

    active_rule4 = np.fmax(clubs_many_level,
                    np.fmax(clubs_several_level,
                        np.fmax(goals_many_level,
                            np.fmax(goals_moderate_level,
                                np.fmax(avgValue_vrich_level,
                                    np.fmax(avgValue_rich_level,
                                        np.fmax(avgValue_mrich_level,
                                            np.fmax(players_many_level,
                                                np.fmax(players_moderate_level,
                                                    np.fmax(teamsChL_mid_level, 
                                                        np.fmax(teamsEuL_mid_level, teamsCfL_mid_level)))))))))))
   
    active_rule5 = np.fmax(clubs_many_level,
                    np.fmax(goals_many_level,
                        np.fmax(avgValue_vrich_level, 
                            np.fmax(players_many_level,
                                np.fmax(teamsChL_max_level, 
                                    np.fmax(teamsEuL_max_level, teamsCfL_min_level)))))) 

    score_activation_tragic = np.fmin(active_rule1, score_tragic)
    score_activation_bad = np.fmin(active_rule2, score_bad)
    score_activation_moderate = np.fmin(active_rule3, score_moderate)
    score_activation_good = np.fmin(active_rule4, score_good)
    score_activation_best = np.fmin(active_rule5, score_best)

    aggregated_rules = np.fmax(score_activation_tragic, np.fmax(score_activation_bad, np.fmax(score_activation_moderate, np.fmax(score_activation_good, score_activation_best))))

    # active_rule1 = np.fmax(qual_level_lo, serv_level_lo)
    # tip_activation_lo = np.fmin(active_rule1, tip_lo)  # removed entirely to 0
    #
    #
    # tip_activation_md = np.fmin(serv_level_md, tip_md)
    #
    # active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
    # tip_activation_hi = np.fmin(active_rule3, tip_hi)
    #
    # aggregated_rules = np.fmax(tip_activation_lo, np.fmax(tip_activation_md, tip_activation_hi))

    score = fuzz.defuzz(unv_score, aggregated_rules, 'centroid')
    score_activation = fuzz.interp_membership(unv_score, aggregated_rules, score)  # for plot
    return score_activation


if __name__ == "__main__":
    initFuzzyMembershipFunctions()
    data_source = None
    while not(data_source == '1' or data_source == '2'):
        data_source = input('Wpisz 1 by wczytać dane z pliku lub 2 by wpisać dane z konsoli:\n')
    if data_source == '1':
        leagues_data = get_leagues_data('leagues.csv')
        for league in leagues_data:
            activateFuzzyMembershipFunctions(league)
            result = defineRules()
            print(f"{league['leagueName']} score is: {result}")
    else:
        league = {'leagueName': input('Podaj nazwę ligi:\n'), 'country': input('Podaj kraj ligi:\n'),
                  'clubs': input('Podaj liczbę klubów grających w lidze:\n'),
                  'goalsPerMatch': input('Podaj średnią liczbę goli na mecz:\n'),
                  'avgMarketValueInMln': input('Podaj średnią wartośc klubu w lidze (w milionach):\n'),
                  'teamsInChampionsLeague': input('Podaj liczbę klubów grających w Lidze Mistrzów:\n'),
                  'teamsInEuropaLeague': input('Podaj liczbę klubów grających w Lidze Europy:\n'),
                  'teamsInConferenceLeague': input('Podaj liczbę klubów grających w Lidze Konferencji:\n'),
                  'famousPlayers': input('Podaj liczbę sławnych piłkarzy:\n')}
        activateFuzzyMembershipFunctions(league)
        result = defineRules()
        print(f"{league['leagueName']} score is: {result}")

    #plt.show()