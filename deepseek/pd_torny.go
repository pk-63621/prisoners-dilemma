package main

import (
	"flag"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type Prisoner struct {
	Name               string
	Strategy           *Strategy
	JailTime           int
	OpponentSum        int
	OpponentDecisions  []Action
	Decisions          []Action
	StrategyLocalState LocalState
}

func NewPrisoner(name string, strategy *Strategy, iterations int) *Prisoner {
	return &Prisoner{
		Name:               name,
		Strategy:           strategy,
		Decisions:          make([]Action, 0, iterations),
		OpponentDecisions:  make([]Action, 0, iterations),
		StrategyLocalState: LocalState{},
	}
}

func (p *Prisoner) addPlay(decision Action, play int) {
	p.JailTime += play
	p.Decisions = append(p.Decisions, decision)
}

func (p *Prisoner) opponentHistory(opponentDecision Action, lastPlay int) {
	p.OpponentSum += lastPlay
	p.OpponentDecisions = append(p.OpponentDecisions, opponentDecision)
}

func (p *Prisoner) getDecision(rng *rand.Rand) Action {
	return p.Strategy.Action(p.Decisions, p.OpponentDecisions, &p.StrategyLocalState, rng)
}

type PrisonersDilemma struct {
	PlayMatrix       [2][2][2]int
	Prisoners        []*Prisoner
	NoiseProbability float64
	RNG              *rand.Rand
}

func NewPrisonersDilemma(playMatrix [2][2][2]int, prisoners []*Prisoner, noiseProbability float64, rng *rand.Rand) *PrisonersDilemma {
	return &PrisonersDilemma{
		PlayMatrix:       playMatrix,
		Prisoners:        prisoners,
		NoiseProbability: noiseProbability,
		RNG:              rng,
	}
}

func (pd *PrisonersDilemma) noiseOccurred() bool {
	return pd.RNG.Float64() < pd.NoiseProbability
}

func (pd *PrisonersDilemma) playNextIteration() ([2]Action, [2]int) {
	decisions := [2]Action{}
	for i, p := range pd.Prisoners {
		decisions[i] = p.getDecision(pd.RNG)
		if pd.noiseOccurred() {
			decisions[i] = complementAction(decisions[i])
		}
	}

	results := pd.PlayMatrix[decisions[0]][decisions[1]]
	for i, p := range pd.Prisoners {
		p.addPlay(decisions[i], results[i])
		p.opponentHistory(decisions[1-i], results[1-i])
	}

	return decisions, results
}

type TournamentParticipant struct {
	Name     string
	Strategy *Strategy
}

func (tp *TournamentParticipant) replicate(generation int) *TournamentParticipant {
	return &TournamentParticipant{
		Name:     fmt.Sprintf("%s.%d", tp.Name, generation),
		Strategy: tp.Strategy,
	}
}

type TournamentParticipantResults struct {
	Scores map[*TournamentParticipant]int
	mu     sync.Mutex
}

func NewTournamentParticipantResults() *TournamentParticipantResults {
	return &TournamentParticipantResults{
		Scores: make(map[*TournamentParticipant]int),
	}
}

func (tpr *TournamentParticipantResults) addScore(participant *TournamentParticipant, score int) {
	tpr.mu.Lock()
	defer tpr.mu.Unlock()
	tpr.Scores[participant] += score
}

type PrisonersDilemmaTournament struct {
	PlayMatrix          [2][2][2]int
	Participants        []*TournamentParticipant
	ParticipantsPerGame int
	Iterations          int
	NoiseErrorProb      float64
	RNG                 *rand.Rand
	Combinations        [][]int
}

func NewPrisonersDilemmaTournament(playMatrix [2][2][2]int, participants []*TournamentParticipant, participantsPerGame, iterations int, noiseErrorProb float64, rng *rand.Rand) *PrisonersDilemmaTournament {
	var combinations [][]int
	if participantsPerGame == 2 {
		combinations = generatePairs(len(participants))
	} else {
		combinations = generateCombinations(len(participants), participantsPerGame)
	}
	return &PrisonersDilemmaTournament{
		PlayMatrix:          playMatrix,
		Participants:        participants,
		ParticipantsPerGame: participantsPerGame,
		Iterations:          iterations,
		NoiseErrorProb:      noiseErrorProb,
		RNG:                 rng,
		Combinations:        combinations,
	}
}

func (pdt *PrisonersDilemmaTournament) playGame(participantIndices []int, rng *rand.Rand) []struct {
	Index int
	Score int
} {
	participants := make([]*TournamentParticipant, len(participantIndices))
	for i, idx := range participantIndices {
		participants[i] = pdt.Participants[idx]
	}

	prisoners := make([]*Prisoner, len(participants))
	for i, part := range participants {
		prisoners[i] = NewPrisoner(fmt.Sprintf("prisoner%d.aka.%s", i+1, part.Name), part.Strategy, pdt.Iterations)
	}

	game := NewPrisonersDilemma(pdt.PlayMatrix, prisoners, pdt.NoiseErrorProb, rng)
	for i := 0; i < pdt.Iterations; i++ {
		game.playNextIteration()
	}

	results := make([]struct {
		Index int
		Score int
	}, len(participantIndices))
	for i, p := range prisoners {
		results[i] = struct {
			Index int
			Score int
		}{Index: participantIndices[i], Score: p.JailTime}
	}

	return results
}

func (pdt *PrisonersDilemmaTournament) playTournament() *TournamentParticipantResults {
	outcome := NewTournamentParticipantResults()
	var wg sync.WaitGroup
	resultsChan := make(chan []struct {
		Index int
		Score int
	}, len(pdt.Combinations))

	for _, combo := range pdt.Combinations {
		wg.Add(1)
		go func(c []int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(pdt.RNG.Int63()))
			res := pdt.playGame(c, rng)
			resultsChan <- res
		}(combo)
	}

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	for res := range resultsChan {
		for _, r := range res {
			outcome.addScore(pdt.Participants[r.Index], r.Score)
		}
	}

	return outcome
}

func generatePairs(n int) [][]int {
	pairs := make([][]int, 0, n*(n-1)/2)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			pairs = append(pairs, []int{i, j})
		}
	}
	return pairs
}

func generateCombinations(n, k int) [][]int {
	if k != 2 {
		return generateCombinationsRecursive(make([]int, n), k)
	}
	return generatePairs(n)
}

func generateCombinationsRecursive(indices []int, k int) [][]int {
	if k == 0 {
		return [][]int{{}}
	}
	if len(indices) == 0 {
		return nil
	}
	head := indices[0]
	tail := indices[1:]
	combinationsWithHead := generateCombinationsRecursive(tail, k-1)
	for i := range combinationsWithHead {
		combinationsWithHead[i] = append([]int{head}, combinationsWithHead[i]...)
	}
	combinationsWithoutHead := generateCombinationsRecursive(tail, k)
	return append(combinationsWithHead, combinationsWithoutHead...)
}

func main() {
	iterations := flag.Int("i", 1000, "Number of iterations per game")
	rounds := flag.Int("r", 5, "Number of rounds of evolution")
	noiseProb := flag.Float64("noise", 0, "Probability of noise (decision flip)")
	flag.Parse()

	playMatrix := [2][2][2]int{
		{{3, 3}, {0, 5}},
		{{5, 0}, {1, 1}},
	}

	participants := []*TournamentParticipant{
		{Name: "p1", Strategy: strategyGandhi()},
		{Name: "p2", Strategy: strategyDefector()},
		{Name: "p3", Strategy: strategyAlternator()},
		{Name: "p4", Strategy: strategyHateOpponent()},
		{Name: "p5", Strategy: strategyGrudger()},
		{Name: "p6", Strategy: strategyAngryGrudger()},
		{Name: "p7", Strategy: strategyRandom()},
		{Name: "p8", Strategy: strategySophist()},
		{Name: "p9", Strategy: strategySuspiciousSophist()},
		{Name: "p10", Strategy: strategyTitForTat()},
		{Name: "p11", Strategy: strategySuspiciousTitForTat()},
		{Name: "p12", Strategy: strategyForgivingTitForTat()},
		{Name: "p13", Strategy: strategyFirmButFair()},
		{Name: "p14", Strategy: strategyPavlov()},
		{Name: "p15", Strategy: strategySuspiciousPavlov()},
		{Name: "p16", Strategy: strategySpookyPavlov()},
		{Name: "p17", Strategy: strategySuspiciousSpookyPavlov()},
		{Name: "p18", Strategy: strategyTwoTitsForTat()},
		{Name: "p19", Strategy: strategySuspiciousTwoTitsForTat()},
		{Name: "p20", Strategy: strategyHardTitForTat()},
		{Name: "p21", Strategy: strategySoftGrudger()},
		{Name: "p22", Strategy: strategyHardMajority()},
		{Name: "p23", Strategy: strategyProber()},
		{Name: "p24", Strategy: strategyHandshake()},
	}

	cumulativeScores := make(map[*TournamentParticipant]int)
	rand.Seed(time.Now().UnixNano())

	for round := 1; round <= *rounds; round++ {
		fmt.Printf("=== Round %d ===\n", round)
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		tournament := NewPrisonersDilemmaTournament(playMatrix, participants, 2, *iterations, *noiseProb, rng)
		results := tournament.playTournament()

		for participant, score := range results.Scores {
			cumulativeScores[participant] += score
		}

		fmt.Println("Results:")
		fmt.Println("-------------------")
		for participant, score := range results.Scores {
			fmt.Printf("Participant: %s, Strategy: %s, Score: %d\n", participant.Name, participant.Strategy.Name, score)
		}
		fmt.Println()
	}

	fmt.Println("=== Final Results ===")
	fmt.Println("---------------------")
	var winner *TournamentParticipant
	maxScore := -1
	for participant, score := range cumulativeScores {
		fmt.Printf("Participant: %s, Strategy: %s, Cumulative Score: %d\n", participant.Name, participant.Strategy.Name, score)
		if score > maxScore {
			maxScore = score
			winner = participant
		}
	}
	fmt.Println("---------------------")
	fmt.Printf("Winner: %s (Strategy: %s) with a cumulative score of %d\n", winner.Name, winner.Strategy.Name, maxScore)
}
