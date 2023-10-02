class HMM:
    """
        @param transModel: {(St-1, St): p}
            The lookup table for the transition model with (S*S) entries,
            where S is the number of possibilites for the state variable.

            St-1 is a value for the state variable at time t-1
            St is a value for the state variable at time t
            p is the corresponding probability

        @param sensorModel: {(e1..eN, si): p}
            The lookup table for the sensor model with (S*E) entries,
            where S is the number of possibilites for the state variable,
            and E is the number of possibilities for the evidence variable.

            e1...eN are the values for evidence variables E1...EN.
            si is the value for the state variable.    
            p is the probability of this combination of values.

        @param priorModel: {S0: p}
            The lookup table for the prior distribution of the state variable, with S entries
            where S is the number of possibilites for the state variable.

            S0 is a value for the state variable at time 0
            p is the corresponding probability
        
        @param stateVariable: Enum
            An enum class to describe the state variable for the HMM. 

        @param stateCardinality: Int
            The number of possibilities for the state variable.
    """
    def __init__(self, transTable, sensorTable, priorTable, stateVariable, stateCardinality):
        self.transTable = transTable
        self.sensorTable = sensorTable
        self.priorTable = priorTable
        self.stateVariable = stateVariable
        self.stateCardinality = stateCardinality


