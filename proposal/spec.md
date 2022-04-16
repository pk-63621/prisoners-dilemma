### Problem Statement
Implement Prisoner's Dilemma for a population including pairs, reproduction and death. The population should follow evolutionary models to refect the real life factor of models.

### Implemetation specification

* Person - gender, (fitness, age), name
* Population - consists of persons
* Strategy - Strat name and rule sets, memory element
* Tournament - Each strat playing against each other, finally result in a winner
* Evolution - Each strat(Pair of persons) at the start, each epoch reproduce or death -- Finally observe the population

#### Rule set

* Reproduction - probalistic chance of each gender child, 2 male cannot reproduce
* Death - Low score or low fitness could kill
* Fitness - Decrease fitness of loners, age could impact it
* If a strat is last, male sacrifice for woman, so from pair male will die -- killing that strat with time since even after winning the female cannot reproduce, no cheating allowed. Maybe multiple pairs can be used for reproduction here.
