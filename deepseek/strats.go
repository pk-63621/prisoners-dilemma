package main

import (
	"math/rand"
)

type Action byte

// type Action int

const (
	Cooperating Action = iota
	Defecting
)

func complementAction(a Action) Action {
	if a == Cooperating {
		return Defecting
	}
	return Cooperating
}

type Strategy struct {
	Name   string
	Action func([]Action, []Action, *LocalState, *rand.Rand) Action
}

func NewStrategy(name string, action func([]Action, []Action, *LocalState, *rand.Rand) Action) *Strategy {
	return &Strategy{Name: name, Action: action}
}

type LocalState struct {
	CntDef           int
	CntCoop          int
	HasDef           bool
	Pending          *int
	NextDecisions    []Action
	StartupDecisions []Action
	FixState         bool
}

func strategyGandhi() *Strategy {
	return NewStrategy("gandhi", func(_, _ []Action, _ *LocalState, _ *rand.Rand) Action {
		return Cooperating
	})
}

func strategyDefector() *Strategy {
	return NewStrategy("defector", func(_, _ []Action, _ *LocalState, _ *rand.Rand) Action {
		return Defecting
	})
}

func strategyAlternator() *Strategy {
	return NewStrategy("alternator", func(ownDecisions, _ []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}
		return complementAction(ownDecisions[len(ownDecisions)-1])
	})
}

func strategyHateOpponent() *Strategy {
	return NewStrategy("hate-opponent", func(_, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(opponentDecisions) == 0 {
			return Defecting
		}
		return complementAction(opponentDecisions[len(opponentDecisions)-1])
	})
}

func strategyGrudger() *Strategy {
	return NewStrategy("grudger", func(ownDecisions, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}

		if ls.HasDef || (len(opponentDecisions) > 0 && opponentDecisions[len(opponentDecisions)-1] == Defecting) {
			ls.HasDef = true
			return Defecting
		}
		return Cooperating
	})
}

func strategyAngryGrudger() *Strategy {
	return NewStrategy("angry-grudger", func(_, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(opponentDecisions) == 0 {
			return Defecting
		}

		if ls.HasDef || (len(opponentDecisions) > 0 && opponentDecisions[len(opponentDecisions)-1] == Defecting) {
			ls.HasDef = true
			return Defecting
		}
		return Cooperating
	})
}

func strategyRandom() *Strategy {
	return NewStrategy("random", func(_, _ []Action, _ *LocalState, rng *rand.Rand) Action {
		if rng.Intn(2) == 0 {
			return Cooperating
		}
		return Defecting
	})
}

func strategySophist() *Strategy {
	return NewStrategy("sophist", func(ownDecisions, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}

		cntCoop, cntDef := getCoopAndDefectCount(ls, opponentDecisions)
		if cntDef > cntCoop {
			return Defecting
		}
		return Cooperating
	})
}

func strategySuspiciousSophist() *Strategy {
	return NewStrategy("suspicious-sophist", func(_, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(opponentDecisions) == 0 {
			return Defecting
		}

		cntCoop, cntDef := getCoopAndDefectCount(ls, opponentDecisions)
		if cntDef > cntCoop {
			return Defecting
		}
		return Cooperating
	})
}

func strategyTitForTat() *Strategy {
	return NewStrategy("tit-for-tat", func(ownDecisions, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}
		if len(opponentDecisions) > 0 {
			return opponentDecisions[len(opponentDecisions)-1]
		}
		return Cooperating
	})
}

func strategySuspiciousTitForTat() *Strategy {
	return NewStrategy("suspicious-tit-for-tat", func(ownDecisions, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Defecting
		}
		if len(opponentDecisions) > 0 {
			return opponentDecisions[len(opponentDecisions)-1]
		}
		return Defecting
	})
}

func strategyForgivingTitForTat() *Strategy {
	return NewStrategy("forgiving-tit-for-tat", func(ownDecisions, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}

		if len(opponentDecisions) >= 2 &&
			opponentDecisions[len(opponentDecisions)-1] == opponentDecisions[len(opponentDecisions)-2] {
			return opponentDecisions[len(opponentDecisions)-1]
		}
		return Cooperating
	})
}

func strategyFirmButFair() *Strategy {
	return NewStrategy("firm-but-fair", func(ownDecisions, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}

		if opponentDecisions[len(opponentDecisions)-1] == ownDecisions[len(ownDecisions)-1] {
			return Cooperating
		} else if ls.HasDef || opponentDecisions[len(opponentDecisions)-1] == Defecting {
			return Defecting
		}
		return Cooperating
	})
}

func strategyPavlov() *Strategy {
	return NewStrategy("pavlov", func(ownDecisions, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}

		if opponentDecisions[len(opponentDecisions)-1] == Defecting {
			return complementAction(ownDecisions[len(ownDecisions)-1])
		}
		return ownDecisions[len(ownDecisions)-1]
	})
}

func strategySuspiciousPavlov() *Strategy {
	return NewStrategy("suspicious-pavlov", func(ownDecisions, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Defecting
		}

		if opponentDecisions[len(opponentDecisions)-1] == Defecting {
			return complementAction(ownDecisions[len(ownDecisions)-1])
		}
		return ownDecisions[len(ownDecisions)-1]
	})
}

func strategySpookyPavlov() *Strategy {
	return NewStrategy("spooky-pavlov", func(ownDecisions, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}

		if opponentDecisions[len(opponentDecisions)-1] != ownDecisions[len(ownDecisions)-1] {
			return complementAction(ownDecisions[len(ownDecisions)-1])
		}
		return ownDecisions[len(ownDecisions)-1]
	})
}

func strategySuspiciousSpookyPavlov() *Strategy {
	return NewStrategy("suspicious-spooky-pavlov", func(ownDecisions, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Defecting
		}

		if opponentDecisions[len(opponentDecisions)-1] != ownDecisions[len(ownDecisions)-1] {
			return complementAction(ownDecisions[len(ownDecisions)-1])
		}
		return ownDecisions[len(ownDecisions)-1]
	})
}

func strategyTwoTitsForTat() *Strategy {
	return NewStrategy("two-tits-for-tat", func(ownDecisions, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Cooperating
		}

		if ls.Pending != nil {
			ls.Pending = nil
			if opponentDecisions[len(opponentDecisions)-1] == Defecting {
				ls.Pending = new(int)
				*ls.Pending = 1
			}
			return Defecting
		}

		if opponentDecisions[len(opponentDecisions)-1] == Defecting {
			ls.Pending = new(int)
			*ls.Pending = 1
			return Defecting
		}
		return opponentDecisions[len(opponentDecisions)-1]
	})
}

func strategySuspiciousTwoTitsForTat() *Strategy {
	return NewStrategy("suspicious-two-tits-for-tat", func(ownDecisions, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			return Defecting
		}

		if len(opponentDecisions) == 1 {
			return opponentDecisions[len(opponentDecisions)-1]
		}

		if ls.Pending != nil {
			ls.Pending = nil
			if opponentDecisions[len(opponentDecisions)-1] == Defecting {
				ls.Pending = new(int)
				*ls.Pending = 1
			}
			return Defecting
		}

		if opponentDecisions[len(opponentDecisions)-1] == Defecting {
			ls.Pending = new(int)
			*ls.Pending = 1
			return Defecting
		}
		return opponentDecisions[len(opponentDecisions)-1]
	})
}

func strategyHardTitForTat() *Strategy {
	return NewStrategy("hard-tit-for-tat", func(_, opponentDecisions []Action, _ *LocalState, _ *rand.Rand) Action {
		if len(opponentDecisions) >= 3 && containsDefection(opponentDecisions[len(opponentDecisions)-3:]) {
			return Defecting
		}
		return Cooperating
	})
}

func strategySoftGrudger() *Strategy {
	return NewStrategy("soft-grudger", func(_, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(opponentDecisions) > 0 && opponentDecisions[len(opponentDecisions)-1] == Defecting {
			ls.NextDecisions = []Action{Defecting, Defecting, Defecting, Defecting, Cooperating, Cooperating}
		}

		if len(ls.NextDecisions) > 0 {
			action := ls.NextDecisions[0]
			ls.NextDecisions = ls.NextDecisions[1:]
			return action
		}
		return Cooperating
	})
}

func strategyHardMajority() *Strategy {
	return NewStrategy("hard-majority", func(_, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		cntCoop, cntDef := getCoopAndDefectCount(ls, opponentDecisions)
		if cntCoop > cntDef {
			return Cooperating
		}
		return Defecting
	})
}

func strategyProber() *Strategy {
	return NewStrategy("prober", func(ownDecisions, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			ls.StartupDecisions = []Action{Defecting, Cooperating, Cooperating}
		}

		if len(ls.StartupDecisions) > 0 {
			action := ls.StartupDecisions[0]
			ls.StartupDecisions = ls.StartupDecisions[1:]
			return action
		}

		if ls.FixState {
			return Defecting
		}

		if len(opponentDecisions) == 3 &&
			opponentDecisions[len(opponentDecisions)-1] == opponentDecisions[len(opponentDecisions)-2] &&
			opponentDecisions[len(opponentDecisions)-1] == Cooperating {
			ls.FixState = true
			return Defecting
		}
		return opponentDecisions[len(opponentDecisions)-1]
	})
}

func strategyHandshake() *Strategy {
	return NewStrategy("handshake", func(ownDecisions, opponentDecisions []Action, ls *LocalState, _ *rand.Rand) Action {
		if len(ownDecisions) == 0 {
			ls.StartupDecisions = []Action{Defecting, Cooperating}
		}

		if len(ls.StartupDecisions) > 0 {
			action := ls.StartupDecisions[0]
			ls.StartupDecisions = ls.StartupDecisions[1:]
			return action
		}

		if len(opponentDecisions) >= 2 &&
			opponentDecisions[0] == Defecting &&
			opponentDecisions[1] == Cooperating {
			return Cooperating
		}
		return Defecting
	})
}

func getCoopAndDefectCount(ls *LocalState, opponentDecisions []Action) (int, int) {
	if len(opponentDecisions) == 0 {
		return 0, 0
	}

	lastAction := opponentDecisions[len(opponentDecisions)-1]
	if lastAction == Defecting {
		ls.CntDef++
	} else {
		ls.CntCoop++
	}

	return ls.CntCoop, ls.CntDef
}

func containsDefection(actions []Action) bool {
	for _, a := range actions {
		if a == Defecting {
			return true
		}
	}
	return false
}

var name2strategy = map[string]*Strategy{
	"gandhi":                      strategyGandhi(),
	"defector":                    strategyDefector(),
	"alternator":                  strategyAlternator(),
	"hate-opponent":               strategyHateOpponent(),
	"grudger":                     strategyGrudger(),
	"angry-grudger":               strategyAngryGrudger(),
	"random":                      strategyRandom(),
	"sophist":                     strategySophist(),
	"suspicious-sophist":          strategySuspiciousSophist(),
	"tit-for-tat":                 strategyTitForTat(),
	"suspicious-tit-for-tat":      strategySuspiciousTitForTat(),
	"forgiving-tit-for-tat":       strategyForgivingTitForTat(),
	"firm-but-fair":               strategyFirmButFair(),
	"pavlov":                      strategyPavlov(),
	"suspicious-pavlov":           strategySuspiciousPavlov(),
	"spooky-pavlov":               strategySpookyPavlov(),
	"suspicious-spooky-pavlov":    strategySuspiciousSpookyPavlov(),
	"two-tits-for-tat":            strategyTwoTitsForTat(),
	"suspicious-two-tits-for-tat": strategySuspiciousTwoTitsForTat(),
	"hard-tit-for-tat":            strategyHardTitForTat(),
	"soft-grudger":                strategySoftGrudger(),
	"hard-majority":               strategyHardMajority(),
	"prober":                      strategyProber(),
	"handshake":                   strategyHandshake(),
}

func allStrategies() []*Strategy {
	strategies := make([]*Strategy, 0, len(name2strategy))
	for _, v := range name2strategy {
		strategies = append(strategies, v)
	}
	return strategies
}
