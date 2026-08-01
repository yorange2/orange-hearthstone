using SabberStoneCore.Model;
using SabberStoneCore.Model.Entities;
using SabberStoneBasicAI.Score;

namespace SabberStoneEnv;

/// <summary>
/// Greedy heuristic opponent with selectable strategy. All five SabberStone
/// one-ply lookahead scoring heuristics are exposed: the same clone-process-rate
/// loop, different board-state evaluators.
/// </summary>
public static class GreedyOpponent
{
    /// <summary>Strategy enum matching the protocol command ("midrange", "aggro", etc.).</summary>
    public enum Strategy { Midrange, Aggro, Control, Fatigue, Ramp }

    /// <summary>Scorer factory.</summary>
    public static Score ScorerFor(Strategy s) => s switch
    {
        Strategy.Aggro    => new AggroScore(),
        Strategy.Control  => new ControlScore(),
        Strategy.Fatigue  => new FatigueScore(),
        Strategy.Ramp     => new RampScore(),
        _                 => new MidRangeScore(),
    };

    /// <summary>Parse a strategy name (case-insensitive). Returns Midrange on unknown input.</summary>
    public static Strategy Parse(string name) => name.ToLowerInvariant() switch
    {
        "aggro"    => Strategy.Aggro,
        "control"  => Strategy.Control,
        "fatigue"  => Strategy.Fatigue,
        "ramp"     => Strategy.Ramp,
        _          => Strategy.Midrange,
    };

    /// <summary>Index (into the current player's Options()) of the greedy-best action for the given
    /// strategy.  One-ply lookahead: clone → process each option → rate acting player's board.</summary>
    public static int BestAction(Game game, Strategy strategy = Strategy.Midrange)
    {
        var scorer = ScorerFor(strategy);
        int actingPid = game.CurrentPlayer.PlayerId;
        int n = game.CurrentPlayer.Options().Count;
        int bestIdx = 0, bestRate = int.MinValue;
        for (int i = 0; i < n; i++)
        {
            // resetRandomSeed: false — the default (true) hands every clone a fresh time-based
            // Random, which makes greedy's own choice nondeterministic whenever an option's
            // outcome involves RNG. Inheriting the parent's stream also evaluates all options
            // under the same draw, so the one-ply comparison is apples-to-apples.
            Game clone = game.Clone(resetRandomSeed: false);
            var opts = clone.CurrentPlayer.Options(); // same order as original (same state)
            clone.Process(opts[i]);
            // Rate the acting player's resulting state (CurrentPlayer flips after END_TURN).
            Controller c = clone.CurrentPlayer.PlayerId == actingPid
                ? clone.CurrentPlayer
                : clone.CurrentPlayer.Opponent;
            scorer.Controller = c;
            int rate = scorer.Rate();
            if (rate > bestRate) { bestRate = rate; bestIdx = i; }
        }
        return bestIdx;
    }
}
