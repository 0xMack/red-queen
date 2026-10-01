<script setup lang="ts">
// Chapter body only -- the header, cover, nav, and prev/next come from pages/learn.vue (driven by
// data/learnChapters.ts). Keep a single root element: page transitions require one.
import * as code from "~/data/snippets/measuring-strength"
import * as math from "~/data/math/measuring-strength"
</script>

<template>
  <article class="prose-chapter">
    <p>
      The <NuxtLink to="/learn/self-play">self-play chapter</NuxtLink> ended with a network that learned Checkers from its own games and then
      went <strong>undefeated in 380 games</strong> on the leaderboard. That was good news, and it broke the leaderboard. The next question is
      which way of <em>training</em> a two-player player works best: opponent pools, leagues, training through search. Answering it needs a
      measurement that can still tell a strong player from a stronger one. This chapter builds that measurement. None of it is specific to
      Checkers.
    </p>

    <h2>The yardstick ran out</h2>
    <p>
      The leaderboard used to rank by <strong>points per game</strong>: every entrant plays every other, a win is 1, a draw ½, and the score is
      the average. That's easy to read, and it has three problems once the best players are much stronger than the rest.
    </p>
    <ul>
      <li>
        <strong>It saturates.</strong> The leader scored 0.94, and most of those points came from beating entrants it was always going to beat.
        By the ratings below, a player 200 Elo stronger still -- expected to score 0.76 against the leader itself -- would score about 0.97
        against the same field, where the leader is expected to score 0.91: a big gain in strength, six hundredths of a point. Comparing two training regimes with a score pinned near 1 is like comparing two sprinters by whether they beat
        a jogger.
      </li>
      <li>
        <strong>The field sets the number.</strong> Add three weak entrants and everyone's score goes up. Points per game describes the field
        as much as the player.
      </li>
      <li>
        <strong>The samples were thin and treated as independent.</strong> Twenty games per pairing, all from the standard start position,
        each counted as its own piece of evidence.
      </li>
    </ul>
    <p>
      The self-play experiments ran into this. Training through search (TD-Leaf) lifted every one of five networks, yet against the fixed
      field of opponents the test could only say p = 0.125, and one seed even looked like a loss. Three changes fix it: better games, a better
      scale, and a better question.
    </p>

    <h2>Openings, played in pairs</h2>
    <p>
      Two deterministic players, starting from the same position, play the same game every time. Twenty games between them can be one game
      repeated twenty times, so the sample is much smaller than it looks. (Here, random tie-breaks keep most games distinct, but they all
      still start from one position.) Tournament checkers faced the same problem with strong human players and solved it with the
      <strong>three-move ballot</strong>: the first three moves are drawn from a list of openings known to be fair to both sides.
    </p>
    <p>
      The project builds its own ballot from the same idea. It takes every position three plies from the start (216 of them, counting two
      move orders that reach the same position once) and keeps the ones that material search, looking 8 plies ahead, scores as level. That
      leaves 174. In the other 42, one side loses a man by force, so a game from there says more about the opening than about the players.
    </p>
    <BallotGrid />
    <p>
      Each opening is played <strong>twice, with the colours swapped</strong>. That's a <strong>game pair</strong>. If an opening happens to
      favour Red, it favours Red in both games, so the luck cancels. A pair scores 0, ½, 1, 1½ or 2 points, and the pair, not the single game,
      is the independent piece of evidence. Every comparison that follows counts pairs.
    </p>
    <CodeBlock :snippet="code.ballot" />

    <h2>Elo: a scale for win probability</h2>
    <p>
      The <strong>Elo scale</strong>, invented to rate chess players, turns a difference in strength into a prediction. A player
      <MathTerm id="r-a" tex="R_A" /> against a player <MathTerm id="r-b" tex="R_B" /> is expected to score
    </p>
    <EloCurve />
    <p>
      A 200-point edge expects about 0.76 points per game, 400 points about 0.91. The curve flattens at the top, so each extra 0.01 near 1 costs
      more rating than the last. That flattening is exactly what points per game hides: near the ceiling, big differences in strength become
      small differences in score.
    </p>
    <CodeBlock :snippet="code.expected" />

    <h2>Ratings from a round robin</h2>
    <p>
      A rating comes from results, not from a single game. Fit one rating per player to <em>all</em> the games of the round robin at once.
      Pick the ratings under which every player's <strong>expected</strong> points, summed over its games, match the points it actually
      scored:
    </p>
    <MathFormula :formula="math.fitted" caption="The Bradley–Terry model's maximum-likelihood condition: the ratings that make the observed results most likely." />
    <p>
      This is the <strong>Bradley–Terry model</strong>, found by Newton's method in a handful of steps. It has the property the old score
      lacked: <strong>beating a much weaker opponent earns almost nothing</strong>, because the model already expected it. The leader's
      rating comes from the games it played against the players nearest its own strength. Two details make it robust. One virtual drawn game
      per pairing keeps a perfect score from sending a rating to infinity. Resampling the game pairs and refitting 200 times gives each
      rating a 95% interval. Random is pinned at 0.
    </p>
    <CodeBlock :snippet="code.bradleyTerry" />
    <RatingLadder />
    <p>
      On today's field the two yardsticks agree on the order. The difference is what they depend on. Keep the leader's games exactly as they
      were, but fit the numbers to the strongest eleven entrants only, and its points per game fall from 0.92 to 0.86. Its rating relative to
      Material 4-ply stays put. A rating measures the player; points per game measure the player and whoever else turned up. The intervals
      also say which gaps are real: TD-Leaf searching 4 plies is clear of everything else. The
      <NuxtLink to="/games/checkers">Checkers leaderboard</NuxtLink> now ranks by this rating. When you play there, your own row is a
      <em>performance rating</em>: the rating at which your results against the entrants you played would be expected.
    </p>

    <h2>Is A stronger than B? Ask sequentially</h2>
    <p>
      A leaderboard answers "where does everyone stand". Comparing training regimes asks something narrower: is the network trained this way
      stronger than the one trained that way? The direct answer is to play them against each other. The remaining question is how many games
      to play. Play too few and you get noise. Fix a large number in advance and you waste most of it, because a big difference is obvious
      after a dozen pairs and only a small one needs hundreds.
    </p>
    <p>
      The <strong>sequential probability ratio test</strong> (SPRT; Wald, 1945) decides as it goes. You state two hypotheses: H0, A is no
      stronger (0 Elo), and H1, A is at least elo1 stronger. You also state the error rates you'll accept, α and β. After every pair it
      computes the <strong>log-likelihood ratio</strong>: how much more likely the pairs so far are under H1 than under H0. When that ratio
      crosses a bound, the test stops. This is how chess engines test every change before accepting it. Here it is, played out on simulated
      players:
    </p>
    <ClientOnly>
      <SprtLab />
      <template #fallback>
        <div class="card not-prose my-6 p-6 text-center text-sm text-fg-subtle">Loading the SPRT lab…</div>
      </template>
    </ClientOnly>
    <p>Try these with the lab:</p>
    <ul>
      <li>
        <strong>A clear difference</strong> (+150) is decided at the first look, after about twenty pairs.
      </li>
      <li>
        <strong>Equal players</strong> (0): the test wrongly says "stronger" about 6% of the time, close to the promised 5%. It says "no
        stronger" nearly 90% of the time, and the rest run out of openings.
      </li>
      <li>
        <strong>A real but small difference</strong> (+20) with elo1 = 20 usually runs out of openings. 174 pairs can't resolve a difference
        that small, so the ballot sets the finest distinction a single head-to-head can make: about 50 Elo, the default.
      </li>
      <li>
        <strong>Between 0 and elo1</strong> (+25 with elo1 = 50), either answer is acceptable. The test promises nothing in that range, so
        choose elo1 as the smallest gain worth detecting.
      </li>
    </ul>
    <CodeBlock :snippet="code.sprt" />
    <Callout variant="note" title="The first look waits for 20 pairs">
      The LLR divides by the variance of the pair scores, and after a handful of pairs that estimate can be tiny by luck. In 4,000 simulated
      tests between equal players (elo1 = 50), looking from the tenth pair on said "stronger" 8% of the time, against the promised 5%.
      Looking from the twentieth pair on brought it to 6%. The small excess that remains comes from treating pair scores as normally
      distributed.
    </Callout>

    <h2>What it says about self-play</h2>
    <p>
      Every self-play experiment from the <NuxtLink to="/learn/self-play">previous chapter</NuxtLink> was measured again this way. Each variant
      played its baseline head to head: the same training seed on both sides, both searching 3 plies, one game pair per ballot opening. With five
      seeds that's 870 pairs per comparison. Alongside is what the old field report said.
    </p>
    <H2hResults />
    <ul>
      <li>
        <strong>Training through search is decisively better.</strong> TD-Leaf at 3 plies beat the same networks trained without search by
        <strong>+120 Elo on every seed</strong> (+103 to +127). The sequential test called it after just 25 pairs. The old report's p = 0.125 was
        the yardstick running out, not the effect being in doubt.
      </li>
      <li>
        <strong>Some "no difference" results were differences the field couldn't see.</strong> The old report put a wider layer, a second small
        layer and five times the games all on the same plateau, within a few hundredths of a point of each other. Head to head, the first two
        are stronger on all five seeds: a 64-wide layer beats a 16-wide one by +55 Elo after a million games, and two 32-wide layers beat one 64-wide layer by +36.
        Against the field of material searchers, the differences just didn't show.
      </li>
      <li>
        <strong>The big results stand, and are now sharp.</strong> Two layers of 64 against one: +143 on every seed. A single 192-wide layer
        with the same number of weights: −151. Monte-Carlo returns (λ = 1): −148. Five times the games for the small network: −14, a little
        worse if anything.
      </li>
      <li>
        <strong>Some really are nothing.</strong> The opponent pool is +5 Elo (−6 to +16) over plain self-play, split across seeds. The previous
        chapter's reading still stands: it buys reliability, not strength.
      </li>
    </ul>

    <h2>What it can't tell you</h2>
    <ul>
      <li>
        <strong>Head to head is not the whole story.</strong> Strength isn't guaranteed to be transitive: A can beat B and B beat C while A
        struggles with C. A head-to-head win is a fact about those two players. The leaderboard's round robin, against everyone, is still the
        place to see how a player fares in general.
      </li>
      <li>
        <strong>Five seeds are still five seeds.</strong> The pooled interval counts pairs, so it answers "are <em>these</em> five networks
        stronger than <em>those</em>". A training regime is a distribution over seeds. When every seed agrees, as with TD-Leaf, there's little
        doubt. When they split, only more seeds help, and an exact test over five seeds can't go below p = 0.0625 however many games are played.
      </li>
      <li>
        <strong>Ratings are relative.</strong> Random is 0 by definition, and a rating means something only within the field it was fitted
        to. Adding an entrant can shift everyone.
      </li>
    </ul>

    <h2>Where this goes next</h2>
    <p>
      With a yardstick that doesn't run out, the training regimes themselves can be compared properly. Candidates include leagues of past
      and present networks, prioritising the opponents a network loses to, populations that evolve the training settings while gradients
      train the weights, and search that improves the targets rather than just the moves. Each will face the same question and the same
      test: is it stronger, by how much, and after how many pairs.
    </p>
  </article>
</template>
