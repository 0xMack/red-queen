<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
import type { H2hRow } from "~/components/lab/results/H2hResults.vue"
import * as code from "~/data/snippets/training-regimes"
import * as math from "~/data/math/training-regimes"

// selfplay-v4 (docs/design/0014), `checkers_selfplay_experiment.py h2h --full`: 2 x 64 networks, 200k self-play games,
// each arm's final network against its baseline's from the same seed, 174 ballot openings x 5 seeds, both at 3 plies.
const REGIMES: H2hRow[] = [
  { arm: "only itself", vs: "the pool", what: "pure self-play, no past selves", elo: -26, lo: -37, hi: -16, seeds: [-40, -43, -38, -63, 51] },
  { arm: "a league", vs: "the pool", what: "every past self, not just the last 10", elo: -38, lo: -50, hi: -28, seeds: [-56, -55, -17, -47, -17] },
  { arm: "prioritized", vs: "the pool", what: "the last 10, games to the ones it can't beat", elo: -11, lo: -22, hi: 1, seeds: [-49, -13, 4, -21, 26] },
  { arm: "league, prioritized", vs: "a league", what: "every past self, prioritized", elo: -8, lo: -20, hi: 3, seeds: [4, -29, -18, 0, 1] },
  { arm: "pool 80% of games", vs: "the pool", what: "past selves 80% of the time, not 50%", elo: -36, lo: -47, hi: -24, seeds: [-63, -62, -22, -36, 3] },
]
</script>

<template>
  <article class="prose-chapter">
    <p>
      A network that learns by <NuxtLink to="/learn/self-play">playing itself</NuxtLink> still has to be told <em>which</em> self to play: the
      current one, frozen copies of its past, or something else entirely. Every famous self-play system made a choice here, and the choices
      differ. TD-Gammon played only itself. AlphaStar kept a league of past and present players and fed each the opponents it was losing to.
      This chapter tries those choices on Checkers, plus one that hands the choosing to evolution, and measures each one head to head, the way the
      <NuxtLink to="/learn/measuring-strength">previous chapter</NuxtLink> built.
    </p>

    <h2>Who does a self-play learner play?</h2>
    <p>
      Playing only yourself has a known failure. The network can learn to beat its current self in a way that forgets how to beat an older one,
      and then circle back, like rock, paper, scissors, chasing its own tail instead of improving. The standard fix is to remember the past:
      keep frozen copies and play them too, so whatever it learns has to work against all of them. The details are where regimes differ: how many
      past selves, how often to play them, and which ones.
    </p>
    <CodeBlock :snippet="code.pickOpponent" />

    <h2>A pool of past selves</h2>
    <p>
      The recipe so far freezes the network every 5,000 games into a pool of the last ten, and plays a random member of the pool half the
      time. With a small network (one hidden layer of 16) the previous chapters found the pool made no difference to strength (+5 Elo). Here every
      regime uses the network that worked best, two hidden layers of 64, trained for 200,000 games, five seeds per regime.
    </p>

    <h2>Every past self: a league</h2>
    <p>
      If remembering ten past selves is good, perhaps remembering all of them is better. That's <strong>fictitious self-play</strong>: the
      opponent is drawn from the learner's entire history, which in theory converges to an unexploitable strategy in games where plain self-play
      cycles. Here that's a pool that never drops anyone, 40 past selves by the end.
    </p>

    <h2>Prioritized opponents</h2>
    <p>
      A long history has a cost: most of it is easy. A network that beats its 40,000-game self nine times in ten learns little from playing it.
      AlphaStar's answer was <strong>prioritized fictitious self-play</strong>: keep a running score against each past self, and send the games to
      the ones it doesn't beat yet. A past self the network has forgotten how to beat gets pulled back in automatically.
    </p>
    <MathScope>
      <PfspDemo />
      <MathFormula :formula="math.running" caption="Each pool member's running score follows its recent results: an exponential average over about the last 20 games against it." />
    </MathScope>

    <h2>What the games said</h2>
    <p>
      Each regime's five final networks played the baseline's five, seed for seed, both searching 3 plies, one game pair on every one of the 174
      ballot openings: 870 pairs a comparison.
    </p>
    <H2hResults :rows="REGIMES" title="Opponent regimes, head to head">
      <template #caption>
        Each regime against the arm it changes one thing from ("the pool": the last 10 past selves, half the games), same training seed on both
        sides. The Elo difference pooled over 870 game pairs, with its interval (the shaded bar), and each seed's (the dots).
        <code>jobs/checkers_selfplay_experiment.py h2h --name selfplay-v4 --full</code>.
      </template>
    </H2hResults>
    <ul>
      <li>
        <strong>The pool earns its keep at this size.</strong> With no past selves at all, four of five seeds lost to the pool (−26 Elo). The
        small network couldn't tell; the bigger one can.
      </li>
      <li>
        <strong>More history was worse, not better.</strong> The league that keeps every past self lost on all five seeds (−38), and so did
        playing the pool 80% of the time instead of half (−36, four of five). Late in training, the league network scored 0.64 points per game
        against its past selves, the ten-member pool's network 0.55: the extra games went to opponents it had already left behind.
      </li>
      <li>
        <strong>Prioritizing didn't rescue it.</strong> Weighting the hard past selves was within noise of uniform, in the ten-member pool (−11,
        seeds split) and in the league (−8). Against its last ten selves the network only just comes out ahead on average (about 0.55 points
        per game), so a recent pool may simply not hold an opponent much harder than the rest for prioritization to find.
      </li>
    </ul>
    <Callout variant="finding" title="The recent past is the best teacher here">
      Both directions lost: dropping the past selves, and playing more of them or older ones. Half the games against the ten most recent past
      selves beat every alternative tried. That fits TD(λ) in Checkers: its targets come from its own next positions, and the most informative
      positions are the ones a strong opponent produces, while a little memory of the recent past probably keeps it from drifting. The
      league and prioritized sampling were built for games with real strategy cycles (StarCraft's rock-paper-scissors of unit counters), which
      Checkers at this level doesn't seem to have.
    </Callout>

    <h2>Population-based training</h2>
    <p>
      The opponent regimes keep one learner and change who it plays. A different idea changes the learner's <em>settings</em> as it goes. In
      <strong>population-based training</strong> (Jaderberg et al., 2017), eight learners train side by side, each with its own learning rate,
      λ, pool fraction and prioritization switch. Every 10,000 games they play a round robin. The bottom quarter then copy the weights and
      settings of a random top-quarter member (<em>exploit</em>) and nudge each setting by ×0.8 or ×1.25 (<em>explore</em>). Gradients train
      the weights, and evolution tunes the schedule around them: the same split of jobs this site's evolution and RL chapters have kept apart.
    </p>
    <CodeBlock :snippet="code.pbt" />
    <PbtLineage />
    <p>
      Copying is ruthless: in every one of the five runs, a single original lineage had taken over the whole population by the last round. The
      settings it ended with didn't agree between runs, though. λ ended anywhere from 0.42 to 0.80 and the pool fraction from 0.15 to 0.59,
      which is what drift looks like, not a schedule being discovered. A population of eight costs eight times the games, so the fair comparison
      is the same population with nothing copied: <strong>random search</strong>, keeping whichever member wins the last round robin.
    </p>
    <H2hResults
      :rows="[
        { arm: 'random search', vs: 'the pool', what: '8 learners, sampled settings, best kept · 8× the games', elo: -10, lo: -21, hi: 1, seeds: [-31, -26, 23, -11, -4] },
        { arm: 'population-based', vs: 'random search', what: 'the same 8, with exploit and explore', elo: -1, lo: -13, hi: 11, seeds: [10, 31, -19, 13, -40] },
      ]"
      title="Populations, head to head"
    >
      <template #caption>
        Each population's champion (its last round robin's best member) against the baseline's network, same seed, 174 ballot openings × 5 seeds.
        The trainer's <code>pbt</code>, arms <code>g-rs</code> and <code>g-pbt</code> of <code>selfplay-v4</code>.
      </template>
    </H2hResults>
    <p>
      Neither beat a single learner with the hand-set defaults. Eight times the compute and picking the best of eight came out level with it
      (−10, interval reaching +1), and population-based training was level with random search (−1). Two things explain most of that. First,
      the defaults were already good: none of the regimes above improved on them either, so there was little headroom for a search to find.
      Second, the selection signal was weak. Each round robin gave a member 28 games, and differences of 20 to 40 Elo, the size that matters
      here, take hundreds of game pairs to see. Copying the "winner" of 28 games is mostly copying luck. PBT would need far more evaluation games
      per round, which makes it much more expensive.
    </p>
    <Callout variant="note" title="A negative result worth keeping">
      Population-based training earned its reputation tuning schedules for large networks over long runs, where a bad learning rate costs days.
      Here a run is minutes, the defaults were found by hand, and the evaluation that drives selection is the bottleneck. Making evaluation cheaper
      and sharper, the job of the previous chapter, is what would let it work.
    </Callout>

    <h2>Where this goes next</h2>
    <p>
      Of everything tried in this chapter, the plain recipe won: half the games against the ten most recent past selves, settings left alone.
      The measured gains so far came from what the network is and how it learns (capacity, training through search), not from who it plays.
      The natural next experiments put those together: TD-Leaf from the start rather than as a finishing step, a search-guided policy as well as a
      value (the AlphaZero combination), and a stronger, fixed field of opponents to check that head-to-head gains are gains against everyone.
    </p>
  </article>
</template>
