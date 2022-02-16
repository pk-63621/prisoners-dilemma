class prisoners_dilemma:
    def __init__(self, matrix):
        self.play_matrix = matrix

    def get_result(self, prisoner1, prisoner2):
        try:
            if prisoner1 not in ["cooperating", "defecting"] or prisoner2 not in ["cooperating", "defecting"]:
                print("value mismatch {} {}".format(prisoner1, prisoner2))
                return
            if prisoner1 is "cooperating" and prisoner2 is "cooperating":
                return self.play_matrix[0][0]
            if prisoner1 is "cooperating" and prisoner2 is "defecting":
                return self.play_matrix[0][1]
            if prisoner1 is "defecting" and prisoner2 is "cooperating":
                return self.play_matrix[1][0]
            if prisoner1 is "defecting" and prisoner2 is "defecting":
                return self.play_matrix[1][1]
        except:
            print("pay matrix must be not correct")


class prisoner:
    def __init__(self, name):
        self.name = name
        self.jail_time = []
        self.opponent_sum = 0
        self.opponent_decisions = []
        self.decisions = []
        self.strategy = None

    def add_play(self, decision, play):
        try:
            if play == None or not decision:
                print("No value for required variables: {} {}".format(play, decision))
                return
            self.jail_time.append(int(play))
            self.decisions.append(decision)
        except Exception as e:
            print("fix the play value", e)

    def opponent_history(self, opponent_decision, last_play):
        try:
            if last_play == None or not opponent_decision:
                print("No value for last_play")
                return
            self.opponent_sum += int(last_play)
            self.opponent_decisions.append(opponent_decision)
        except Exception as e:
            print("Fix opponent history", e)

    def get_result(self):
        try:
            return sum(self.jail_time)
        except:
            print("Must be some non interger values: {}".format(self.jail_time))


    def set_strategy(self, name):
        self.strategy = name

    def get_decision(self):
        if self.strategy == "defector":
            return "defecting"
        if self.strategy == "idiot":
            return "cooperating"
        if self.strategy == "tit for tat":
            if len(self.opponent_decisions) > 0:
                return self.opponent_decisions[-1]
            return "cooperating"
        return "cooperating"


play_matrix = [[(1, 1), (3, 0)],[(0, 3), (2, 2)]]
game = prisoners_dilemma(play_matrix)
prisoner1 = prisoner("prisoner1") 
prisoner1.set_strategy("idiot")
prisoner2 = prisoner("prisoner2")
prisoner2.set_strategy("defector")
for i in range(10):
    prisoner1_decision = prisoner1.get_decision()
    prisoner2_decision = prisoner2.get_decision()
    prisoner1_play, prisoner2_play = game.get_result(prisoner1_decision, prisoner2_decision)
    print("Game: {} {}".format(prisoner1_decision, prisoner2_decision))
    print("Result: {} {}".format(prisoner1_play, prisoner2_play))
    print()

    # adding reward info
    prisoner1.add_play(prisoner1_decision, prisoner1_play)
    prisoner1.opponent_history(prisoner2_decision, prisoner2_play)
    prisoner2.add_play(prisoner2_decision, prisoner2_play)
    prisoner2.opponent_history(prisoner1_decision, prisoner1_play)

print(prisoner1.get_result())
print(prisoner2.get_result())
