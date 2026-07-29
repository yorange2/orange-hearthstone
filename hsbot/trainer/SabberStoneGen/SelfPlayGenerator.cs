using System.Text.Json;
using SabberStoneCore.Config;
using SabberStoneCore.Enums;
using SabberStoneCore.Model;
using SabberStoneCore.Tasks.PlayerTasks;

namespace SabberStoneGen;

/// <summary>
/// Generates value-net training data by self-play. For each game it snapshots the acting
/// player's observable board at the moment they choose to END their turn (the capture point
/// in FEATURES.md §3 — before EndTurnTask resolves), then labels every snapshot with whether
/// that player went on to win.
///
/// Behavior policy v1: uniform-random over legal options. This is the simplest valid data
/// source (matches SabberStone's own examples); a stronger policy (greedy heuristic, or
/// epsilon-greedy over the value net) will yield higher-quality data later.
/// </summary>
public sealed class SelfPlayGenerator
{
    private static readonly CardClass[] Classes =
    {
        CardClass.DRUID, CardClass.HUNTER, CardClass.MAGE, CardClass.PALADIN, CardClass.PRIEST,
        CardClass.ROGUE, CardClass.SHAMAN, CardClass.WARLOCK, CardClass.WARRIOR,
    };

    private readonly SabberStoneObserver _observer = new();
    private static readonly JsonSerializerOptions Json = new() { IncludeFields = false };

    /// <summary>Plays <paramref name="numGames"/> games, writing one JSONL row per snapshot.</summary>
    /// <returns>rows written.</returns>
    public long Generate(int numGames, int seed, TextWriter writer)
    {
        var rnd = new Random(seed);
        long rows = 0;

        for (int g = 0; g < numGames; g++)
        {
            var game = new Game(new GameConfig
            {
                StartPlayer = rnd.Next(1, 3),
                Player1HeroClass = Classes[rnd.Next(Classes.Length)],
                Player2HeroClass = Classes[rnd.Next(Classes.Length)],
                FillDecks = true,
                Shuffle = true,
                SkipMulligan = true,
                Logging = false,
                History = false,
            });
            game.StartGame();

            var snapshots = new List<(float[] Feats, int PlayerId)>();

            while (game.State != State.COMPLETE)
            {
                var options = game.CurrentPlayer.Options();
                if (options.Count == 0) break; // defensive; should not happen while RUNNING
                var task = options[rnd.Next(options.Count)];

                if (task is EndTurnTask)
                {
                    snapshots.Add((FeatureExtractor.Extract(_observer.Observe(game)), game.CurrentPlayer.PlayerId));
                }

                game.Process(task);
            }

            int? winner = game.Player1.PlayState == PlayState.WON ? 1
                        : game.Player2.PlayState == PlayState.WON ? 2
                        : (int?)null;
            if (winner is null) continue; // skip ties / unresolved

            foreach (var (feats, playerId) in snapshots)
            {
                var row = new Row { Label = playerId == winner ? 1 : 0, Features = feats };
                writer.WriteLine(JsonSerializer.Serialize(row, Json));
                rows++;
            }
        }

        return rows;
    }

    private sealed class Row
    {
        public int Label { get; set; }
        public float[] Features { get; set; } = Array.Empty<float>();
    }
}
