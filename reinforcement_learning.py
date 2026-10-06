class ReinforcementLearning:
    def __init__(self):
        self.q_table = {}

    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {
                "accion_1": 0,
                "accion_2": 0,
                "accion_3": 0
            }

        return max(self.q_table[state], key=self.q_table[state].get)