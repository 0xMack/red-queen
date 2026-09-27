<script setup lang="ts">
import { BANDIT_RESULTS } from "~/data/banditResults"

// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.

const updateCode = `fn update(&mut self, row: usize, arm: usize, reward: f64) {
    let c = self.cell(row, arm);
    self.counts[c] += 1;
    let step = if self.alpha > 0.0 {
        self.alpha                         // a constant step: recent payouts count more
    } else {
        1.0 / self.counts[c] as f64        // 1/n: exactly the running average
    };
    self.values[c] += step * (reward - self.values[c]);   // move toward what just happened
}`

const epsilonCode = `fn choose(&mut self, row: usize, rng: &mut Rng) -> usize {
    if rng.uniform() < self.epsilon() {
        return rng.below(self.arms as u32) as usize;         // explore: any machine
    }
    argmax_random(self.values(row), rng)                   // exploit: the best one so far
}`

const ucbCode = `// untried machines first; then the estimate plus a bonus for how little it has been tried
self.c * self.scale * sqrt(ln(t) / n)`

const thompsonCode = `/// A draw from Beta(a, b) for whole numbers a, b: the a-th smallest of a + b - 1 uniforms.
pub fn beta_order_statistic(a: u32, b: u32, rng: &mut Rng) -> f64 {
    let mut draws: Vec<f64> = (0..a + b - 1).map(|_| rng.uniform()).collect();
    draws.sort_by(|x, y| x.partial_cmp(y).unwrap());
    draws[a as usize - 1]
}
// each pull: draw from Beta(1 + wins, 1 + losses) for every machine, pull the highest draw`
</script>

<template>
  <article class="prose-chapter">
    <p>
      Five slot machines stand in a row. Each pays out at its own rate, and nobody tells you what those rates are. You
      have a hundred pulls. Every pull you spend finding out how good a machine is is a pull you didn't spend on the best
      machine you know of -- and every pull you spend on the best machine you know of is a chance you didn't take to find
      a better one. That tension has a name, <strong>exploration versus exploitation</strong>, and it runs through every
      learning algorithm in the rest of this part.
    </p>
    <p>
      This is a <strong>multi-armed bandit</strong> ("one-armed bandit" being an old name for a slot machine). It is
      reinforcement learning with everything else stripped away: there is one situation, nothing you do changes what
      comes next, and the only question is which lever to pull. It's where the ideas of the next chapters are easiest to
      see -- and, at the end, it's one small step from a Q-table.
    </p>

    <h2>One situation, five choices</h2>
    <p>
      You can play it before reading on: the <NuxtLink to="/games/bandit?mode=play">Bandit game</NuxtLink> puts you
      against strategies playing the <em>same</em> five machines. Everything here runs on one Rust implementation (the
      same code the leaderboard's evaluation runs, compiled to WebAssembly for this page), and two of its properties
      make the comparisons fair:
    </p>
    <ul>
      <li>
        <strong>The same seed is the same game, and the same luck.</strong> Every machine draws its payouts from its own
        random stream, so the fifth pull of machine C pays the same whoever pulls it and whatever they pulled before.
      </li>
      <li>
        <strong>The score measures choices, not dice.</strong> <em>Skill</em> compares the average value of the machines
        you pulled with pulling at random (0) and with pulling the best machine every time (100). A lucky run on a bad
        machine doesn't count as skill.
      </li>
    </ul>

    <h2>Keeping score: a table with one row</h2>
    <p>
      Anything that plays this game has to keep some record of what each machine has paid. The simplest is an
      <strong>estimate</strong> per machine: the average of its payouts so far. There is a way to keep an average without
      storing every payout -- nudge the old estimate toward each new payout by a step of 1/n:
    </p>
    <CodeBlock lang="rust" :code="updateCode" />
    <p>
      Five machines, one number each: that's a table with <strong>one row and five columns</strong>. Hold on to that
      picture. The Q-table in the next chapter is exactly this, with a row for every situation instead of one row -- and
      the update above, with a step of α instead of 1/n, <em>is</em> Q-learning's update in a world with no next state.
    </p>

    <h2>Greedy, and why it fails</h2>
    <p>
      The obvious strategy is <strong>greedy</strong>: always pull the machine with the best estimate. It fails in a way
      that's easy to watch. All estimates start at 0; greedy pulls a machine, it pays once, and that machine now leads. If
      that machine is decent it will keep paying often enough to stay ahead, and greedy never tries the others again.
      Watch the pull tape: a greedy game is a few scattered pulls and then one colour to the end.
    </p>
    <BanditLab title="A strategy playing, and the table it keeps" :scenarios="['classic', 'lucky-start', 'close-call']" strategy="greedy" />
    <p>
      On the classic game (five win-or-lose machines, one clearly best) greedy spends only <strong>38%</strong> of its
      pulls on the best machine, for a skill of <strong>39</strong>. When the payouts are noisy (<em>Lucky start</em>),
      its first lucky draw decides everything, and its skill is <strong>0.5</strong>: no better than pulling at random.
      The information greedy lacks isn't hard to get. It just never pays for it.
    </p>

    <h2>Exploring on purpose</h2>
    <p>
      Every other strategy here is a different answer to <em>how much</em> to explore and <em>when</em>. The oldest is
      <strong>ε-greedy</strong>: be greedy, except that one pull in ten (ε = 0.1) goes to a machine picked at random.
    </p>
    <CodeBlock lang="rust" :code="epsilonCode" />
    <p>
      A fixed ε keeps exploring at the same rate after it has long since found the best machine; a
      <strong>decaying ε</strong> starts curious (0.3) and settles down (to 0 by pull 100). Two strategies explore without
      any randomness at all:
    </p>
    <ul>
      <li>
        <strong>Optimistic start</strong> assumes every machine pays the maximum until it has been tried -- one imaginary
        perfect pull per machine. Untried machines look best, so each gets tried; a machine only keeps its lead by
        actually paying.
      </li>
      <li>
        <strong>UCB</strong> (upper confidence bound) adds to each estimate a bonus that shrinks the more the machine has
        been pulled -- "it could be this good, for all I know" -- and pulls the highest total:
      </li>
    </ul>
    <CodeBlock lang="rust" :code="ucbCode" />
    <p>
      And one strategy explores by keeping a whole <em>belief</em> per machine. For a win-or-lose machine,
      <strong>Thompson sampling</strong> keeps a curve of where its win rate could be -- wide when it has barely been tried,
      a narrow peak once it has -- and each pull draws one plausible rate from every curve and pulls the highest draw. A
      barely-tried machine has a wide curve, so it sometimes draws high and gets tried; a machine known to be poor almost
      never does. (Choose Thompson sampling in the lab above to see the curves.) The draw itself needs nothing fancier than
      sorting some random numbers:
    </p>
    <CodeBlock lang="rust" :code="thompsonCode" />

    <h2>Which strategy wins depends on the game</h2>
    <p>
      The leaderboard ranks strategies on the classic game, but the <NuxtLink to="/games/bandit">game</NuxtLink> has
      six more scenarios, each built to set a particular trap. Here is every strategy on every scenario, 500 games each
      that no setting was tuned on:
    </p>
    <BanditScenarioMatrix :rows="BANDIT_RESULTS" />
    <ul>
      <li>
        <strong>Optimism is the best strategy on the classic game</strong> (78), ahead of Thompson sampling (64). With
        payouts of 0 or 1, "assume every machine is perfect until tried" costs one pull per machine and nothing more.
      </li>
      <li>
        <strong>UCB's textbook constant is for long games.</strong> With the bonus from the theory (c = √2), UCB scores
        39 -- tied with greedy -- because in 100 pulls it keeps re-checking machines it should have given up on. Sized for
        the game (c = 0.5), it scores 71. "Provably optimal as the game goes on forever" and "good at this game" are
        different claims.
      </li>
      <li>
        <strong>Too many arms punishes insisting on trying everything.</strong> With 16 machines, UCB1 must pull every
        one before it compares them, and scores 19 against ε-greedy's 51.
      </li>
      <li>
        <strong>A rare jackpot defeats nearly everything.</strong> One machine pays 50 two percent of the time (worth 1.0
        a pull); another always pays 0.8. Simply pulling the steady machine every time would score about 46. The best
        strategy here manages 31, and the "smart" ones (optimism, UCB, Thompson) manage about 5: they size their
        exploration to the jackpot's enormous spread, and never stop exploring.
      </li>
      <li><strong>Close calls are hard for everyone.</strong> When the best machine is only slightly better, 100 pulls isn't enough to be sure of anything: the best score is 22.</li>
    </ul>
    <Callout variant="finding" title="The design doc was wrong twice">
      Before any of this was measured, <code>docs/design/0011</code> expected Thompson sampling to usually win, and
      proposed a "deceptive optimism" scenario -- every machine poor -- to show optimism failing. Measured, optimism beat
      Thompson on every scenario, and stayed the best strategy even when every machine was poor. The scenario was dropped:
      a lesson the numbers don't show isn't one.
    </Callout>

    <h2>When the world changes</h2>
    <p>
      In <em>Drifting</em>, the best machine breaks somewhere between pulls 50 and 70: from then on it pays like the
      worst, and the runner-up is best. A player who stays loyal to the machine that was best at the start scores
      <strong>-13</strong> -- worse than random. The strategies that keep an average (a step of 1/n) have 60 good pulls of
      evidence for the broken machine; it takes dozens of losses to drag that average below the runner-up's.
    </p>
    <p>
      The fix is the step size. With a <strong>constant step</strong> (0.2), each new payout moves an estimate a fixed
      fraction of the way, so old evidence fades: ε-greedy rises from 43 to <strong>55</strong>. It pays for it on the
      steady games (52 instead of 57 on the classic one), because a constant step never settles -- the same trade-off every
      learning rate makes. Optimism is no help here (49): it explores once, at the start, and never again. UCB, which never
      entirely stops exploring, does best (59).
    </p>
    <BanditLab title="When the best machine breaks" :scenarios="['drifting']" strategy="epsilon-tracking" />

    <h2>Letting evolution choose the settings</h2>
    <p>
      Every strategy above has knobs, and every number in the table depends on how they were set. So let evolution set
      them. <code>jobs/bandit_evolve_run.py</code> treats ε-greedy's four settings -- how often it explores, how quickly
      that exploring fades, how far each payout moves an estimate, and what an untried machine is assumed to pay -- as a
      genome of four numbers, and evolves a population of 32 of them for 40 generations, scoring each on 100 fresh
      training games every generation (none of them from the table's held-out games).
    </p>
    <p>
      Evolution's answer, on the classic game: explore at random almost never (ε 0.001), start every machine at 0.59 --
      a little above what an average machine pays -- and move estimates slowly (step 0.05). It turned ε-greedy into
      <strong>optimistic start</strong>, the strategy that was already winning, and tuned it: <strong>80</strong> on unseen
      games, the best in the table. The same settings also lead on <em>Drifting</em> (67) and <em>Jackpot</em> (42) --
      a slow constant step both forgets a broken machine and keeps a rare jackpot from being written off after one miss.
    </p>
    <p>
      And they fail completely on <em>Lucky start</em> (5), whose machines pay 4 to 7 on average: starting every machine
      at 0.59 isn't optimistic there, it's pessimistic, and the strategy becomes greedy. Evolving on four win-or-lose
      scenarios at once gives a generalist (the best on <em>Close call</em> and <em>Drifting</em>) with exactly the same
      blind spot. Evolution finds the best settings for the games it's shown -- and says nothing about the rest. Both
      runs are on the <NuxtLink to="/runs">runs page</NuxtLink>, with the settings they evolved generation by
      generation.
    </p>

    <h2>Two lamps: when the situation matters</h2>
    <p>
      Now a lamp above the machines lights red or blue before each pull, and the machines pay differently under each
      colour: the best machine under red is the worst under blue. Averaged over both colours, every machine pays the same.
    </p>
    <p>
      A strategy that can't see the lamp keeps one row of values, so it averages the two situations -- and finds nothing:
      every strategy scores about <strong>0</strong>, however clever. Let the same strategies see the lamp, and each keeps
      <strong>two rows</strong>, one per colour: Thompson sampling goes from 0 to 51, optimism to 68. Nothing about the
      algorithms changed. What changed is <em>what they could observe</em>, and so how many rows their table has.
    </p>
    <BanditLab title="One row or two: seeing the lamp" :scenarios="['two-lamps']" strategy="thompson" lamp :initial-lamp="false" />
    <p>
      This is the step from a bandit to reinforcement learning proper -- a <strong>contextual bandit</strong>: the best
      action depends on the situation, so the table gets a row per situation. Snake's Q-table in the next chapter has a row
      for every combination of its 11 yes-or-no features: 2<sup>11</sup> = 2,048 rows. And the failure is the one that
      stops it there: when two situations need different moves but <em>look</em> the same, they share one row, and no
      amount of learning can make that row right for both.
    </p>

    <h2>The detour: when a pull changes what comes next</h2>
    <p>
      One more change turns the lamp into something bigger. In <em>Detour</em> the lamp isn't a coin toss any more: it's
      the <strong>room</strong> you're in, and your pull decides the next one. In the red room, one machine -- the detour --
      pays nothing at all, but lights the gold room, where every machine pays up to 3 (1.2 to 2.4 on average) and leads
      back to red. Detour, then the best gold machine, earns about 1.2 a pull; the best red machine, 0.75.
    </p>
    <p>
      Every strategy so far judges a machine by what it pays. By that measure the detour is the worst machine in the
      room, and none of them ever takes it on purpose: all score <strong>10 or less</strong>. Even pulling the best red
      machine perfectly, from the first pull to the last, would score only 21. What's missing is a way to value a pull
      for <em>where it leads</em>:
    </p>
    <p class="text-center font-mono text-sm text-fg">Q(room, machine) ← Q + α · [ payout + γ · max Q(next room, ·) − Q ]</p>
    <p>
      That is the Q-learning update, and the table is the same two-row table as <em>Two lamps</em>: a row per room. γ
      (gamma) says how much the next room's value counts. At γ = 0 it's the bandit update and the detour looks worthless;
      raise it and the detour's value fills in from the gold room behind it, although the detour itself never pays. The
      lab puts γ on a dial -- watch the red room's row:
    </p>
    <BanditLab title="Detour: valuing where a pull leads" :scenarios="['detour']" gamma />
    <p>
      Measured on 500 games: γ 0 scores <strong>6</strong>, like every other strategy; γ 0.5 scores 28; γ 0.9 scores
      <strong>55</strong>, and takes the best move 52% of the time. And γ 0.99 scores <strong>-1</strong>: this agent starts
      optimistic (every value 10), and with the future counting almost fully those optimistic values echo back and forth
      between the rooms faster than 100 pulls can wear them down. Looking ahead is only useful at the right distance.
    </p>

    <h2>From here to Q-learning</h2>
    <p>
      That update <em>is</em> Q-learning. What separates this game from Snake is only size: two rooms here, 2,048
      situations there, each a row of the same table and each move a step from one row to the next.
    </p>
    <p>
      The Q-learning agent from the next chapter plays every scenario above, unchanged. With γ 0 it is ε-greedy with a
      constant step, and scores 31 on the classic game; looking ahead, it wins the detour and is ordinary everywhere else. It was built for games with a future, where
      steady exploration over hundreds of thousands of moves pays off; in a 100-pull game it's one strategy among many,
      and not a good one. The gradient bandit, which learns <em>preferences</em> and nudges them by how much a payout beat
      the average so far, is the other thread to follow: that is exactly REINFORCE with a baseline, in one row, and it
      comes back in the policy-gradients chapter.
    </p>
  </article>
</template>
