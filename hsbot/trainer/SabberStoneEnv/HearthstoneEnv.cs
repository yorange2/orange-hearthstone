using SabberStoneCore.Config;
using SabberStoneCore.Enums;
using SabberStoneCore.Model;
using SabberStoneCore.Model.Entities; // Controller

namespace SabberStoneEnv;

/// <summary>
/// Single-agent RL environment over SabberStone (learner = player 1; a uniform-random
/// opponent plays player 2 inside the env). Control returns only at learner decision points:
///
///   Reset()    -> sets Tokens / Privileged / LegalActionFeatures
///   Step(idx)  -> (reward, done); updates the same properties
///
/// State is an entity-token SET (TokenEncoder, "v2") for the transformer policy — not the flat
/// FeatureExtractor vector. Reward is terminal only: +1 win, -1 loss, 0 tie.
/// </summary>
public sealed class HearthstoneEnv
{
    public const int TokenDim = TokenEncoder.Dim;      // 18 (per entity)
    public const int ActDim = ActionEncoder.Dim;       // 20 (per legal action)
    public const int PrivDim = PrivilegedEncoder.Dim;  // 8  (critic-only, training)

    private static readonly CardClass[] Classes =
    {
        CardClass.MAGE, CardClass.HUNTER, CardClass.WARRIOR, CardClass.PALADIN,
    };

    private readonly Random _rnd;
    private const int LearnerId = 1;
    private const int MaxDecisions = 500; // safety cap; HS games terminate via fatigue anyway

    private Game _game = null!;
    private List<SabberStoneCore.Tasks.PlayerTasks.PlayerTask> _legal = new();
    private int _decisions;

    public HearthstoneEnv(int seed) => _rnd = new Random(seed);

    /// <summary>Entity tokens for the current decision point: [numTokens, TokenDim].</summary>
    public float[][] Tokens { get; private set; } = System.Array.Empty<float[]>();

    /// <summary>Legal-action features: [numActions, ActDim].</summary>
    public float[][] LegalActionFeatures { get; private set; } = System.Array.Empty<float[]>();

    /// <summary>Privileged (opponent-hidden) features for the critic only: [PrivDim].</summary>
    public float[] Privileged { get; private set; } = new float[PrivDim];

    public void Reset()
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
        Observe();
    }

    /// <summary>Apply the chosen legal action; returns (reward, done). Reads Tokens/etc. after.</summary>
    public (float reward, bool done) Step(int actionIndex)
    {
        if (actionIndex < 0 || actionIndex >= _legal.Count)
            actionIndex = 0; // defensive; caller should only pass legal indices

        _game.Process(_legal[actionIndex]);
        _decisions++;
        AdvanceToLearnerOrEnd();

        if (_game.State == State.COMPLETE || _decisions >= MaxDecisions)
        {
            Tokens = System.Array.Empty<float[]>();
            LegalActionFeatures = System.Array.Empty<float[]>();
            Privileged = new float[PrivDim];
            return (TerminalReward(), true);
        }

        Observe();
        return (0f, false);
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

    private void Observe()
    {
        _legal = _game.CurrentPlayer.Options();
        var feats = new float[_legal.Count][];
        Controller me = _game.CurrentPlayer;
        for (int i = 0; i < _legal.Count; i++)
            feats[i] = ActionEncoder.Encode(_legal[i], me);
        LegalActionFeatures = feats;
        Privileged = PrivilegedEncoder.Encode(_game.CurrentOpponent);
        Tokens = TokenEncoder.Encode(_game);
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
