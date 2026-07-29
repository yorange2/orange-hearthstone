using SabberStoneCore.Config;
using SabberStoneCore.Enums;
using SabberStoneCore.Model;
using SabberStoneCore.Model.Entities; // Controller
using SabberStoneGen; // FeatureExtractor, SabberStoneObserver (state features + contract)

namespace SabberStoneEnv;

/// <summary>
/// Single-agent RL environment over SabberStone: the learning agent controls player 1; the
/// opponent (player 2) plays a fixed uniform-random policy inside the env. Control returns to
/// the caller only at the learner's decision points, so from the outside it's a clean MDP:
///
///   Reset()      -> Observation (state features + legal-action features)
///   Step(idx)    -> Observation', reward, done
///
/// Reward is terminal only: +1 win, -1 loss, 0 tie. Observations are the 144-float contract
/// vector (from the learner's perspective); actions are ActionEncoder vectors, one per legal
/// PlayerTask. Upgrade paths: self-play opponent / opponent pool, reward shaping.
/// </summary>
public sealed class HearthstoneEnv
{
    public const int ObsDim = FeatureExtractor.Length; // 144
    public const int ActDim = ActionEncoder.Dim;       // 20

    private static readonly CardClass[] Classes =
    {
        CardClass.MAGE, CardClass.HUNTER, CardClass.WARRIOR, CardClass.PALADIN,
    };

    private readonly SabberStoneObserver _observer = new();
    private readonly Random _rnd;
    private const int LearnerId = 1;
    private const int MaxDecisions = 500; // safety cap; HS games terminate via fatigue anyway

    private Game _game = null!;
    private List<SabberStoneCore.Tasks.PlayerTasks.PlayerTask> _legal = new();
    private int _decisions;

    public HearthstoneEnv(int seed) => _rnd = new Random(seed);

    /// <summary>Legal-action feature matrix for the current decision point: [numActions, ActDim].</summary>
    public float[][] LegalActionFeatures { get; private set; } = System.Array.Empty<float[]>();

    public float[] Reset()
    {
        _game = new Game(new GameConfig
        {
            StartPlayer = _rnd.Next(1, 3),
            Player1HeroClass = Classes[_rnd.Next(Classes.Length)],
            Player2HeroClass = Classes[_rnd.Next(Classes.Length)],
            FillDecks = true,
            Shuffle = true,
            SkipMulligan = true,
            Logging = false,
            History = false,
        });
        _game.StartGame();
        _decisions = 0;
        AdvanceToLearnerOrEnd();
        return Observe();
    }

    /// <summary>Apply the chosen legal action; returns (observation, reward, done).</summary>
    public (float[] obs, float reward, bool done) Step(int actionIndex)
    {
        if (actionIndex < 0 || actionIndex >= _legal.Count)
            actionIndex = 0; // defensive; caller should only pass legal indices

        _game.Process(_legal[actionIndex]);
        _decisions++;
        AdvanceToLearnerOrEnd();

        if (_game.State == State.COMPLETE || _decisions >= MaxDecisions)
            return (new float[ObsDim], TerminalReward(), true);

        return (Observe(), 0f, false);
    }

    // Play random opponent moves until it's the learner's turn again or the game ends.
    private void AdvanceToLearnerOrEnd()
    {
        while (_game.State == State.RUNNING && _game.CurrentPlayer.PlayerId != LearnerId)
        {
            var opts = _game.CurrentPlayer.Options();
            if (opts.Count == 0) break;
            _game.Process(opts[_rnd.Next(opts.Count)]);
        }
    }

    private float[] Observe()
    {
        _legal = _game.CurrentPlayer.Options();
        var feats = new float[_legal.Count][];
        Controller me = _game.CurrentPlayer;
        for (int i = 0; i < _legal.Count; i++)
            feats[i] = ActionEncoder.Encode(_legal[i], me);
        LegalActionFeatures = feats;
        return FeatureExtractor.Extract(_observer.Observe(_game));
    }

    private float TerminalReward()
    {
        var learner = _game.Player1.PlayerId == LearnerId ? _game.Player1 : _game.Player2;
        return learner.PlayState switch
        {
            PlayState.WON => 1f,
            PlayState.LOST or PlayState.CONCEDED => -1f,
            _ => 0f,
        };
    }
}
