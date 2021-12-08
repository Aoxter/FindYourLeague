import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from read_data import get_leagues_data

# https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html

# Variables for universes
unv_clubs = unv_goals = unv_avgValue = unv_totValue = unv_players = unv_teamsChL = unv_teamsEuL = unv_teamsCfL = unv_score = None

# Variables for fuzzy membership functions
clubs_low = clubs_mid = clubs_hig = None
goals_low = goals_mid = goals_hig = None
avgValue_low = avgValue_mid = avgValue_hig = None
totValue_low = totValue_mid = totValue_hig = None
players_low = players_mid = players_hig = None
teamsChL_low = teamsChL_mid = teamsChL_hig = None
teamsEuL_low = teamsEuL_mid = teamsEuL_hig = None
teamsCfL_low = teamsCfL_mid = teamsCfL_hig = None
score_low = score_mid = score_hig = None

# Activations of fuzzy membership functions
clubs_low_level = clubs_mid_level = clubs_hig_level = None
goals_low_level = goals_mid_level = goals_hig_level = None
avgValue_low_level = avgValue_mid_level = avgValue_hig_level = None
totValue_low_level = totValue_mid_level = totValue_hig_level = None
players_low_level = players_mid_level = players_hig_level =  None
teamsChL_low_level = teamsChL_mid_level = teamsChL_hig_level = None
teamsEuL_low_level = teamsEuL_mid_level = teamsEuL_hig_level = None
teamsCfL_low_level = teamsCfL_mid_level = teamsCfL_hig_level = None


def initFuzzyMembershipFunctions():
    global unv_clubs, unv_goals, unv_avgValue, unv_totValue, unv_players, unv_teamsChL, unv_teamsEuL, unv_teamsCfL, unv_score
    global clubs_low, clubs_mid, clubs_hig
    global goals_low, goals_mid, goals_hig
    global avgValue_low, avgValue_mid, avgValue_hig
    global totValue_low, totValue_mid, totValue_hig
    global players_low, players_mid, players_hig
    global teamsChL_low, teamsChL_mid, teamsChL_hig
    global teamsEuL_low, teamsEuL_mid, teamsEuL_hig
    global teamsCfL_low, teamsCfL_mid, teamsCfL_hig
    global score_low, score_mid, score_hig

    # Universes for variables
    unv_clubs = np.arange(2, 31, 1)
    unv_goals = np.arange(2, 4, 0.01)
    unv_avgValue = np.arange(0.01, 500.0, 0.01)
    unv_totValue = np.arange(0.01, 10000.0, 0.01)
    unv_players = np.arange(0, 100, 1)
    unv_teamsChL = np.arange(0, 10, 1)
    unv_teamsEuL = np.arange(0, 10, 1)
    unv_teamsCfL = np.arange(0, 10, 1)
    # Result score universum
    unv_score = np.arange(0, 100, 1)

    # Generate fuzzy membership functions
    # trimf - triangular function
    # trapmf - trapezoidal function
    clubs_low = fuzz.trapmf(unv_clubs, [2, 2, 10, 14])
    clubs_mid = fuzz.trapmf(unv_clubs, [10, 12, 16, 18])
    clubs_hig = fuzz.trapmf(unv_clubs, [14, 16, 30, 30])

    goals_low = fuzz.trapmf(unv_goals, [2.0, 2.0, 2.5, 3.0])
    goals_mid = fuzz.trapmf(unv_goals, [2.0, 2.5, 3.0, 3.5])
    goals_hig = fuzz.trapmf(unv_goals, [3.0, 3.5, 4.0, 4.0])

    avgValue_low = fuzz.trapmf(unv_avgValue, [0.01, 0.01, 100.0, 150.0])
    avgValue_mid = fuzz.trapmf(unv_avgValue, [100.0, 150.0, 350.0, 400.0])
    avgValue_hig = fuzz.trapmf(unv_avgValue, [350.0, 400.0, 500.0, 500.0])

    totValue_low = fuzz.trapmf(unv_totValue, [0.01, 0.01, 1000.0, 2000.0])
    totValue_mid = fuzz.trapmf(unv_totValue, [1000.0, 2000.0, 7000.0, 8000.0])
    totValue_hig = fuzz.trapmf(unv_totValue, [7000.0, 8000.0, 10000.0, 10000.0])

    players_low = fuzz.trapmf(unv_players, [0, 0, 5, 10])
    players_mid = fuzz.trapmf(unv_players, [5, 10, 15, 20])
    players_hig = fuzz.trapmf(unv_players, [15, 20, 100, 100])

    teamsChL_low = fuzz.trapmf(unv_teamsChL, [0, 0, 1, 2])
    teamsChL_mid = fuzz.trimf(unv_teamsChL, [1, 2, 3])
    teamsChL_hig = fuzz.trapmf(unv_teamsChL, [2, 3, 10, 10])

    teamsEuL_low = fuzz.trapmf(unv_teamsEuL, [0, 0, 1, 2])
    teamsEuL_mid = fuzz.trimf(unv_teamsEuL, [1, 2, 3])
    teamsEuL_hig = fuzz.trapmf(unv_teamsEuL, [2, 3, 10, 10])

    teamsCfL_low = fuzz.trapmf(unv_teamsCfL, [0, 0, 1, 2])
    teamsCfL_mid = fuzz.trimf(unv_teamsCfL, [1, 2, 3])
    teamsCfL_hig = fuzz.trapmf(unv_teamsCfL, [2, 3, 10, 10])

    score_low = fuzz.trapmf(unv_score, [0, 0, 20, 40])
    score_mid = fuzz.trapmf(unv_score, [20, 40, 60, 80])
    score_hig = fuzz.trapmf(unv_score, [60, 80, 100, 100])


def activateFuzzyMembershipFunctions(values):
    global unv_clubs, unv_goals, unv_avgValue, unv_totValue, unv_players, unv_teamsChL, unv_teamsEuL, unv_teamsCfL
    global clubs_low, clubs_mid, clubs_hig
    global goals_low, goals_mid, goals_hig
    global avgValue_low, avgValue_mid, avgValue_hig
    global totValue_low, totValue_mid, totValue_hig
    global players_low, players_mid, players_hig
    global teamsChL_low, teamsChL_mid, teamsChL_hig
    global teamsEuL_low, teamsEuL_mid, teamsEuL_hig
    global teamsCfL_low, teamsCfL_mid, teamsCfL_hig
    global clubs_low_level, clubs_mid_level, clubs_hig_level
    global goals_low_level, goals_mid_level, goals_hig_level
    global avgValue_low_level, avgValue_mid_level, avgValue_hig_level
    global totValue_low_level, totValue_mid_level, totValue_hig_level
    global players_low_level, players_mid_level, players_hig_level
    global teamsChL_low_level, teamsChL_mid_level, teamsChL_hig_level
    global teamsEuL_low_level, teamsEuL_mid_level, teamsEuL_hig_level
    global teamsCfL_low_level, teamsCfL_mid_level, teamsCfL_hig_level

    # Activation of fuzzy membership functions at given values.
    clubs_low_level = fuzz.interp_membership(unv_clubs, clubs_low, values['clubs'])
    clubs_mid_level = fuzz.interp_membership(unv_clubs, clubs_mid, values['clubs'])
    clubs_hig_level = fuzz.interp_membership(unv_clubs, clubs_hig, values['clubs'])

    goals_low_level = fuzz.interp_membership(unv_goals, goals_low, values['goalsPerMatch'])
    goals_mid_level = fuzz.interp_membership(unv_goals, goals_mid, values['goalsPerMatch'])
    goals_hig_level = fuzz.interp_membership(unv_goals, goals_hig, values['goalsPerMatch'])

    avgValue_low_level = fuzz.interp_membership(unv_avgValue, avgValue_low, values['avgMarketValueInMln'])
    avgValue_mid_level = fuzz.interp_membership(unv_avgValue, avgValue_mid, values['avgMarketValueInMln'])
    avgValue_hig_level = fuzz.interp_membership(unv_avgValue, avgValue_hig, values['avgMarketValueInMln'])

    totValue_low_level = fuzz.interp_membership(unv_totValue, totValue_low, values['totalMarketValueInMln'])
    totValue_mid_level = fuzz.interp_membership(unv_totValue, totValue_mid, values['totalMarketValueInMln'])
    totValue_hig_level = fuzz.interp_membership(unv_totValue, totValue_hig, values['totalMarketValueInMln'])

    players_low_level = fuzz.interp_membership(unv_players, players_low, values['famousPlayers'])
    players_mid_level = fuzz.interp_membership(unv_players, players_mid, values['famousPlayers'])
    players_hig_level = fuzz.interp_membership(unv_players, players_hig, values['famousPlayers'])

    teamsChL_low_level = fuzz.interp_membership(unv_teamsChL, teamsChL_low, values['teamsInChampionsLeague'])
    teamsChL_mid_level = fuzz.interp_membership(unv_teamsChL, teamsChL_mid, values['teamsInChampionsLeague'])
    teamsChL_hig_level = fuzz.interp_membership(unv_teamsChL, teamsChL_hig, values['teamsInChampionsLeague'])

    teamsEuL_low_level = fuzz.interp_membership(unv_teamsEuL, teamsEuL_low, values['teamsInEuropaLeague'])
    teamsEuL_mid_level = fuzz.interp_membership(unv_teamsEuL, teamsEuL_mid, values['teamsInEuropaLeague'])
    teamsEuL_hig_level = fuzz.interp_membership(unv_teamsEuL, teamsEuL_hig, values['teamsInEuropaLeague'])

    teamsCfL_low_level = fuzz.interp_membership(unv_teamsCfL, teamsCfL_low, values['teamsInConferenceLeague'])
    teamsCfL_mid_level = fuzz.interp_membership(unv_teamsCfL, teamsCfL_mid, values['teamsInConferenceLeague'])
    teamsCfL_hig_level = fuzz.interp_membership(unv_teamsCfL, teamsCfL_hig, values['teamsInConferenceLeague'])


def defineRules():
    global clubs_low_level, clubs_mid_level, clubs_hig_level
    global goals_low_level, goals_mid_level, goals_hig_level
    global avgValue_low_level, avgValue_mid_level, avgValue_hig_level
    global totValue_low_level, totValue_mid_level, totValue_hig_level
    global players_low_level, players_mid_level, players_hig_level
    global teamsChL_low_level, teamsChL_mid_level, teamsChL_hig_level
    global teamsEuL_low_level, teamsEuL_mid_level, teamsEuL_hig_level
    global teamsCfL_low_level, teamsCfL_mid_level, teamsCfL_hig_level
    global score_low, score_mid, score_hig, unv_score

    # OR
    active_rule1 = np.fmax(clubs_low_level,
                                   np.fmax(goals_low_level,
                                           np.fmax(avgValue_low_level,
                                                   np.fmax(totValue_low_level,
                                                           np.fmax(players_low_level,
                                                                   np.fmax(teamsChL_low_level,
                                                                           np.fmax(teamsEuL_low_level, teamsCfL_low_level)))))))

    active_rule2 = np.fmax(clubs_mid_level,
                                   np.fmax(goals_mid_level,
                                           np.fmax(avgValue_mid_level,
                                                   np.fmax(totValue_mid_level,
                                                           np.fmax(players_mid_level,
                                                                   np.fmax(teamsChL_mid_level,
                                                                           np.fmax(teamsEuL_mid_level, teamsCfL_mid_level)))))))

    active_rule3 = np.fmax(clubs_hig_level,
                                   np.fmax(goals_hig_level,
                                           np.fmax(avgValue_hig_level,
                                                   np.fmax(totValue_hig_level,
                                                           np.fmax(players_hig_level,
                                                                   np.fmax(teamsChL_hig_level,
                                                                           np.fmax(teamsEuL_hig_level, teamsCfL_hig_level)))))))

    score_activation_low = np.fmin(active_rule1, score_low)
    score_activation_mid = np.fmin(active_rule2, score_mid)
    score_activation_hig = np.fmin(active_rule3, score_hig)

    aggregated_rules = np.fmax(score_activation_low, np.fmax(score_activation_mid, score_activation_hig))

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
    leagues_data = get_leagues_data()
    for league in leagues_data:
        activateFuzzyMembershipFunctions(league)
        result = defineRules()
        print(f"{league['leagueName']} score is: {result}")

    #plt.show()