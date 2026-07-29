using SabberStoneCore.Model;
using SabberStoneCore.Model.Entities;

namespace SabberStoneEnv;

/// <summary>
/// A greedy heuristic opponent built from SabberStone's OWN baseline scoring
/// (<c>SabberStoneBasicAI.Score.MidRangeScore</c>): one-ply lookahead — for each legal option,
/// clone the game, apply it, and rate the acting player's resulting board; pick the best.
///
/// This is a hand-tuned-heuristic benchmark — the SabberStone-side analogue of Hearthstone-
/// Script's `基础策略` (also a greedy weighted heuristic). Evaluating our RL policy against it
/// gives an honest "do we beat a heuristic?" number instead of "do we beat random?".
/// </summary>
public static class GreedyOpponent
{
    private static readonly SabberStoneBasicAI.Score.Score Scorer = new SabberStoneBasicAI.Score.MidRangeScore();

    /// <summary>Index (into the current player's Options()) of the greedy-best action.</summary>
    public static int BestAction(Game game)
    {
        int actingPid = game.CurrentPlayer.PlayerId;
        int n = game.CurrentPlayer.Options().Count;
        int bestIdx = 0, bestRate = int.MinValue;
        for (int i = 0; i < n; i++)
        {
            Game clone = game.Clone();
            var opts = clone.CurrentPlayer.Options(); // same order as original (same state)
            clone.Process(opts[i]);
            // Rate the acting player's resulting state (CurrentPlayer flips after END_TURN).
            Controller c = clone.CurrentPlayer.PlayerId == actingPid
                ? clone.CurrentPlayer
                : clone.CurrentPlayer.Opponent;
            Scorer.Controller = c;
            int rate = Scorer.Rate();
            if (rate > bestRate) { bestRate = rate; bestIdx = i; }
        }
        return bestIdx;
    }
}
