import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from read_data import get_leagues_data
import pandas as pd

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
        self.rule1 = ctrl.Rule((self.goals['few'] | self.avgValue['vpoor'] | self.avgValue['poor']) & self.players['none'] & self.teamsCfL['max'] | self.teamsChL['min'], self.score['tragic'])
        self.rule2 = ctrl.Rule((self.goals['few'] | self.goals['moderate']) & (self.avgValue['mpoor'] | self.avgValue['poor']) & (self.players['none'] | self.players['few']) & self.teamsCfL['max'] | self.teamsEuL['mid'], self.score['bad'])
        self.rule3 = ctrl.Rule((self.avgValue['mpoor'] | self.avgValue['moderate'] | self.avgValue['mrich']) & (self.players['few'] | self.players['moderate']) & (self.teamsEuL['mid'] | self.teamsEuL['max']) & (self.teamsChL['min'] | self.teamsChL['mid']), self.score['moderate'])
        self.rule4 = ctrl.Rule((self.clubs['many'] | self.clubs['several']) & (self.goals['many'] | self.goals['moderate']) & (self.avgValue['mrich'] | self.avgValue['rich']) & (self.players['many'] | self.players['moderate']) | self.teamsChL['mid'] | self.teamsEuL['mid'] | self.teamsCfL['mid'], self.score['good'])
        self.rule5 = ctrl.Rule((self.clubs['many'] | self.goals['many'] | self.avgValue['rich'] | self.avgValue['vrich']) & (self.players['many']) & (self.teamsChL['max']), self.score['best'])

        self.league_ctrl = ctrl.ControlSystem([self.rule1, self.rule2, self.rule3, self.rule4, self.rule5])
        self.league_system = ctrl.ControlSystemSimulation(self.league_ctrl)

    def getScore(self, values):
        self.league_system.input['clubs'] = int(values['clubs'])
        self.league_system.input['goals'] = float(values['goalsPerMatch'])
        self.league_system.input['avgValue'] = float(values['avgMarketValueInMln'])
        self.league_system.input['players'] = int(values['famousPlayers'])
        self.league_system.input['teamsChL'] = int(values['teamsInChampionsLeague'])
        self.league_system.input['teamsEuL'] = int(values['teamsInEuropaLeague'])
        self.league_system.input['teamsCfL'] = int(values['teamsInConferenceLeague'])

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
        data_source = input('Wpisz 1 by wczyta?? dane z pliku lub 2 by wpisa?? dane z konsoli:\n')
    if data_source == '1':
        leagues_data = get_leagues_data('leagues.csv')
        for league in leagues_data:
            #activateFuzzyMembershipFunctions(league)
            #result = defineRules()
            result = controller.getScore(league)
            print(f"{league['leagueName']} score is: {result}")
            league_name_col.append(league['leagueName'])
            league_score_col.append(result)
        data_save = input('Wpisz 1 by zapisa?? dane do pliku lub dowolny klawisz ??eby kontynuowa??:\n')
        if data_save == '1':
            file_name = input('Podaj nazw?? pliku:\n')
            file_name += ".csv"
            d = {'League':league_name_col,'Score':league_score_col}
            df = pd.DataFrame(d)
            df.to_csv(file_name, index=False)
            print('Wyniki zosta??y zapisane w pliku csv w folderze programu')
    else:
        league = {'leagueName': input('Podaj nazw?? ligi:\n'), 'country': input('Podaj kraj ligi:\n'),
                  'clubs': input('Podaj liczb?? klub??w graj??cych w lidze:\n'),
                  'goalsPerMatch': input('Podaj ??redni?? liczb?? goli na mecz:\n'),
                  'avgMarketValueInMln': input('Podaj ??redni?? warto??c klubu w lidze (w milionach):\n'),
                  'teamsInChampionsLeague': input('Podaj liczb?? klub??w graj??cych w Lidze Mistrz??w:\n'),
                  'teamsInEuropaLeague': input('Podaj liczb?? klub??w graj??cych w Lidze Europy:\n'),
                  'teamsInConferenceLeague': input('Podaj liczb?? klub??w graj??cych w Lidze Konferencji:\n'),
                  'famousPlayers': input('Podaj liczb?? s??awnych pi??karzy:\n')}
        # activateFuzzyMembershipFunctions(league)
        # result = defineRules()
        result = controller.getScore(league)
        print(f"{league['leagueName']} score is: {result}")
    view_plot = input('Wpisz 1 je??eli chcesz obejrze?? wykresy funkcji i regu?? dla ostatnio sprawdzanej ligi lub dowolny klawisz ??eby kontynuowa??:\n')
    if view_plot == '1':
        while (function_rule_result != '0'):
            #function_rule_result = input('Wybierz rodzaj wykresu:\n1 - wykresy funkcji przynale??no??ci\n2 - wykresy regu??\n3 - wykres wyniku kontrolera\n0 - sko??cz wizualizacj??\n')
            function_rule_result = input('Wybierz rodzaj wykresu:\n1 - wykresy funkcji przynale??no??ci\n2 - wykres wyniku kontrolera\n0 - sko??cz wizualizacj??\n')
            if function_rule_result == "1":
                while (function_id != '8'):
                    function_id = input('Wybierz funkcje przynale??no??ci:\n0 - funkcje punkt??w atrakcyjno??ci ligi\n1 - funkcje klub??w w lidze\n2 - funkcje ??redniej ilo??ci gol??w\n3 - funkcje ??redniej warto??ci rynkowej\n4 - funkcje ilo??ci s??ynnych zawodnik??w\n5 - funkcje dru??yn w lidze mistrz??w\n6 - funkcj?? dru??yn w lidze europy\n7 - funkcj?? dru??yn w lidze konfederacji\n8 - powr??t do poprzedniego menu\n')
                    if function_id == "0" or function_id == "1" or function_id == "2" or function_id == "3" or function_id == "4" or function_id == "5" or function_id == "6" or function_id == "7":
                        controller.displayMembershipFunctionPlot(int(function_id))
            # elif function_rule_result == "2":
            #     while (rule_id != '6'):
            #         rule_id = input('Wybierz regu????:\n1 - regu??a dla tragicznych lig\n2 - regu??a dla s??abych lig\n3 - regu??a dla przeci??tnych lig\n4 - regu??a dla dobrych lig\n5 - regu??a dla najlepszych lig\n6 - powr??t do poprzedniego menu\n')
            #         if rule_id == "1" or rule_id == "2" or rule_id == "3" or rule_id == "4" or rule_id == "5":
            #             controller.displayRulePlot(int(rule_id))
            elif function_rule_result == "2":
                controller.displayControllerResultPlot()
    #plt.show()