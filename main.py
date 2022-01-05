import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from read_data import get_leagues_data
import pandas as pd

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

    # OR zmienić na AND (bo Or wybija wysokie wartości)
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

class LeagueController:
    clubs = None
    goals = None
    avgValue = None
    players = None
    teamsChL = None
    teamsEuL = None
    teamsCfL = None
    score = None

    rule1 = None
    rule2 = None
    rule3 = None
    rule4 = None
    rule5 = None

    league_ctrl = None
    league_system = None

    def __init__(self):
        # Universes for variables
        self.clubs = ctrl.Antecedent(np.arange(2, 31, 1), 'clubs')
        self.goals = ctrl.Antecedent(np.arange(0.99, 5.01, 0.01), 'goals')
        self.avgValue = ctrl.Antecedent(np.arange(0.01, 500.01, 0.01), 'avgValue')
        self.players = ctrl.Antecedent(np.arange(0, 101, 1), 'players')
        self.teamsChL = ctrl.Antecedent(np.arange(0, 6, 1), 'teamsChL')
        self.teamsEuL = ctrl.Antecedent(np.arange(0, 6, 1), 'teamsEuL')
        self.teamsCfL = ctrl.Antecedent(np.arange(0, 6, 1), 'teamsCfL')
        # Result score universum
        self.score = ctrl.Consequent(np.arange(0, 101, 1), 'score')

        # Membership Functions
        # trimf - triangular function
        # trapmf - trapezoidal function
        self.clubs['few'] = fuzz.trapmf(self.clubs.universe, [2, 2, 5, 8])
        self.clubs['some'] = fuzz.trapmf(self.clubs.universe, [3, 6, 9, 12])
        self.clubs['several'] = fuzz.trapmf(self.clubs.universe, [8, 11, 15, 18])
        self.clubs['many'] = fuzz.trapmf(self.clubs.universe, [15, 18, 30, 30])    

        self.goals['few'] = fuzz.trapmf(self.goals.universe, [1.0, 1.0, 2.0, 3.0])
        self.goals['moderate'] = fuzz.trimf(self.goals.universe, [2.5, 3.0, 3.5])
        self.goals['many'] = fuzz.trapmf(self.goals.universe, [3.0, 3.5, 5.0, 5.0])
        
        self.avgValue['vpoor'] = fuzz.trapmf(self.avgValue.universe, [0.01, 0.01, 1.0, 2.0])
        self.avgValue['poor'] = fuzz.trimf(self.avgValue.universe, [1.0, 5.0, 15.0])
        self.avgValue['mpoor'] = fuzz.trimf(self.avgValue.universe, [7.5, 20.0, 50.0])
        self.avgValue['moderate'] = fuzz.trimf(self.avgValue.universe, [20.0, 50.0, 100.0])
        self.avgValue['mrich'] = fuzz.trimf(self.avgValue.universe, [50.0, 100.0, 175.0])
        self.avgValue['rich'] = fuzz.trimf(self.avgValue.universe, [150.0, 200.0, 275.0])
        self.avgValue['vrich'] = fuzz.trapmf(self.avgValue.universe, [200.0, 300.0, 500.0, 500.0])

        self.players['none'] = fuzz.trimf(self.players.universe, [0, 0, 1])
        self.players['few'] = fuzz.trapmf(self.players.universe, [1, 5, 10, 15])
        self.players['moderate'] = fuzz.trapmf(self.players.universe, [10, 15, 20, 25])
        self.players['many'] = fuzz.trapmf(self.players.universe, [20, 25, 100, 100])

        self.teamsChL['min'] = fuzz.trapmf(self.teamsChL.universe, [0, 0, 1, 2])
        self.teamsChL['mid'] = fuzz.trapmf(self.teamsChL.universe, [1, 2, 3, 4])
        self.teamsChL['max'] = fuzz.trapmf(self.teamsChL.universe, [3, 4, 5, 5])

        self.teamsEuL['min'] = fuzz.trimf(self.teamsEuL.universe, [0, 0, 1])
        self.teamsEuL['mid'] = fuzz.trimf(self.teamsEuL.universe, [0, 1, 2])
        self.teamsEuL['max'] = fuzz.trapmf(self.teamsEuL.universe, [1, 2, 5, 5])

        self.teamsCfL['min'] = fuzz.trapmf(self.teamsCfL.universe, [0, 0, 1, 2])
        self.teamsCfL['mid'] = fuzz.trimf(self.teamsCfL.universe, [1, 2, 3])
        self.teamsCfL['max'] = fuzz.trapmf(self.teamsCfL.universe, [2, 3, 5, 5])

        self.score['tragic'] = fuzz.trapmf(self.score.universe, [0, 0, 15, 25])
        self.score['bad'] = fuzz.trapmf(self.score.universe, [15, 25, 35, 45])
        self.score['moderate'] = fuzz.trapmf(self.score.universe, [35, 45, 55, 65])
        self.score['good'] = fuzz.trapmf(self.score.universe, [55, 65, 75, 85])
        self.score['best'] = fuzz.trapmf(self.score.universe, [75, 85, 100, 100])

        # FuzzyRules
        self.rule1 = ctrl.Rule(self.clubs['some'] | self.clubs['few'] | self.goals['few'] | self.avgValue['mpoor'] | self.avgValue['poor'] | self.avgValue['vpoor'] | self.players['none'] | self.teamsChL['min'] | self.teamsEuL['min'] | self.teamsCfL['max'] | self.teamsCfL['mid'], self.score['tragic'])
        self.rule2 = ctrl.Rule(self.clubs['several'] | self.clubs['some'] | self.clubs['few'] | self.goals['moderate'] | self.goals['few'] | self.avgValue['moderate'] | self.avgValue['mpoor'] | self.avgValue['poor'] | self.players['few'] | self.players['none'] | self.teamsChL['min'] | self.teamsEuL['min'] | self.teamsCfL['max'] | self.teamsCfL['mid'], self.score['bad'])
        self.rule3 = ctrl.Rule(self.clubs['several'] | self.clubs['some'] | self.goals['moderate'] | self.avgValue['mrich'] | self.avgValue['moderate'] | self.avgValue['mpoor'] | self.players['moderate'] | self.players['few'] | self.teamsChL['mid'] | self.teamsChL['min'] | self.teamsEuL['mid'] | self.teamsEuL['min'] | self.teamsCfL['max'] | self.teamsCfL['mid'], self.score['moderate'])
        self.rule4 = ctrl.Rule(self.clubs['many'] | self.clubs['several'] | self.goals['many'] | self.goals['moderate'] | self.avgValue['vrich'] | self.avgValue['rich'] | self.avgValue['mrich'] | self.players['many'] | self.players['moderate'] | self.teamsChL['mid'] | self.teamsEuL['mid'] | self.teamsCfL['mid'], self.score['good'])
        self.rule5 = ctrl.Rule(self.clubs['many'] | self.goals['many'] | self.avgValue['vrich'] | self.players['many'] | self.teamsChL['max'] | self.teamsEuL['max'] | self.teamsCfL['min'], self.score['best'])

        self.league_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4, self.rule5])
        self.league_system = ctrl.ControlSystemSimulation(self.league_ctrl)

    def getScore(self, values):
        self.league_system.input['clubs'] = values['clubs']
        self.league_system.input['goals'] = values['goalsPerMatch']
        self.league_system.input['avgValue'] = values['avgMarketValueInMln']
        self.league_system.input['players'] = values['famousPlayers']
        self.league_system.input['teamsChL'] = values['teamsInChampionsLeague']
        self.league_system.input['teamsEuL'] = values['teamsInEuropaLeague']
        self.league_system.input['teamsCfL'] = values['teamsInConferenceLeague']

        self.league_system.compute()

        return self.league_system.output['score']

        #score.view(sim=league_system)

    def displayMembershipFunctionPlot(self, function_id):
        if function_id == 0:
            self.score.view()
        elif function_id == 1:
            self.clubs.view()
        elif function_id == 2:
            self.goals.view()
        elif function_id == 3:
            self.avgValue.view()
        elif function_id == 4:
            self.players.view()
        elif function_id == 5:
            self.teamsChL.view()
        elif function_id == 6:
            self.teamsEuL.view()
        elif function_id == 7:
            self.teamsCfL.view()

    def displayRulePlot(self, rule_id):
        if rule_id == 1:
            self.rule1.view()
        elif rule_id == 2:
            self.rule2.view()
        elif rule_id == 3:
            self.rule3.view()
        elif rule_id == 4:
            self.rule4.view()
        elif rule_id == 5:
            self.rule5.view()

    def displayControllerResultPlot(self):
        self.score.view(sim=self.league_system)



if __name__ == "__main__":
    league_name_col = []
    league_score_col = []
    # initFuzzyMembershipFunctions()
    controller = LeagueController()
    data_source = None
    function_rule_result = None
    function_id = None
    rule_id = None
    view_plot = None
    data_save = None
    while not(data_source == '1' or data_source == '2'):
        data_source = input('Wpisz 1 by wczytać dane z pliku lub 2 by wpisać dane z konsoli:\n')
    if data_source == '1':
        leagues_data = get_leagues_data('leagues.csv')
        for league in leagues_data:
            #activateFuzzyMembershipFunctions(league)
            #result = defineRules()
            result = controller.getScore(league)
            print(f"{league['leagueName']} score is: {result}")
            league_name_col.append(league['leagueName'])
            league_score_col.append(result)
        data_save = input('Wpisz 1 by zapisać dane do pliku lub dowolny klawisz żeby kontynuować:\n')
        if data_save == '1':
            file_name = input('Podaj nazwę pliku:\n')
            file_name += ".csv"
            d = {'League':league_name_col,'Score':league_score_col}
            df = pd.DataFrame(d)
            df.to_csv(file_name, index=False)
            print('Wyniki zostały zapisane w pliku csv w folderze programu')
    else:
        league = {'leagueName': input('Podaj nazwę ligi:\n'), 'country': input('Podaj kraj ligi:\n'),
                  'clubs': input('Podaj liczbę klubów grających w lidze:\n'),
                  'goalsPerMatch': input('Podaj średnią liczbę goli na mecz:\n'),
                  'avgMarketValueInMln': input('Podaj średnią wartośc klubu w lidze (w milionach):\n'),
                  'teamsInChampionsLeague': input('Podaj liczbę klubów grających w Lidze Mistrzów:\n'),
                  'teamsInEuropaLeague': input('Podaj liczbę klubów grających w Lidze Europy:\n'),
                  'teamsInConferenceLeague': input('Podaj liczbę klubów grających w Lidze Konferencji:\n'),
                  'famousPlayers': input('Podaj liczbę sławnych piłkarzy:\n')}
        # activateFuzzyMembershipFunctions(league)
        # result = defineRules()
        result = controller.getScore(league)
        print(f"{league['leagueName']} score is: {result}")
    view_plot = input('Wpisz 1 jeżeli chcesz obejrzeć wykresy funkcji i reguł dla ostatnio sprawdzanej ligi lub dowolny klawisz żeby kontynuować:\n')
    if view_plot == '1':
        while (function_rule_result != '0'):
            function_rule_result = input('Wybierz rodzaj wykresu:\n1 - wykresy funkcji przynależności\n2 - wykresy reguł\n3 - wykres wyniku kontrolera\n0 - skończ wizualizację\n')
            if function_rule_result == "1":
                while (function_id != '8'):
                    function_id = input('Wybierz funkcje przynależności:\n0 - funkcje punktów atrakcyjności ligi\n1 - funkcje klubów w lidze\n2 - funkcje średniej ilości golów\n3 - funkcje średniej wartości rynkowej\n4 - funkcje ilości słynnych zawodników\n5 - funkcje drużyn w lidze mistrzów\n6 - funkcję drużyn w lidze europy\n7 - funkcję drużyn w lidze konfederacji\n8 - powrót do poprzedniego menu\n')
                    if function_id == "0" or function_id == "1" or function_id == "2" or function_id == "3" or function_id == "4" or function_id == "5" or function_id == "6" or function_id == "7":
                        controller.displayMembershipFunctionPlot(int(function_id))
            elif function_rule_result == "2":
                while (rule_id != '6'):
                    rule_id = input('Wybierz regułę:\n1 - reguła dla tragicznych lig\n2 - reguła dla słabych lig\n3 - reguła dla przeciętnych lig\n4 - reguła dla dobrych lig\n5 - reguła dla najlepszych lig\n6 - powrót do poprzedniego menu\n')
                    if rule_id == "1" or rule_id == "2" or rule_id == "3" or rule_id == "4" or rule_id == "5":
                        controller.displayRulePlot(int(rule_id))
            elif function_rule_result == "3":
                controller.displayControllerResultPlot()
    #plt.show()