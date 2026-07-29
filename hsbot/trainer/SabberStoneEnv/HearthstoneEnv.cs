using SabberStoneCore.Config;
using SabberStoneCore.Enums;
using SabberStoneCore.Model;
using SabberStoneCore.Model.Entities; // Controller

namespace SabberStoneEnv;

/// <summary>
/// Seat-agnostic two-player SabberStone env for self-play. It does NOT play any opponent
/// itself — at every decision point it returns the state from the *current mover's* view and
/// exposes <see cref="CurrentPlayerId"/>; the Python driver routes each decision to the main
/// policy or a league opponent and only trains on the main seat's transitions.
///
///   Reset()    -> sets Tokens / Privileged / LegalActionFeatures / CurrentPlayerId
///   Step(idx)  -> done; on done <see cref="Winner"/> is 1/2/0 (tie). No auto-advance across
///                 players: after a move the next decision (same or other player) is exposed.
///
/// State is an entity-token set (TokenEncoder); `Privileged` is the current mover's opponent-
/// hidden info (critic-only). Reward is derived by the driver from Winner vs the main seat.
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
    private const int MaxDecisions = 800; // safety cap; HS games terminate via fatigue anyway

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

    /// <summary>PlayerId (1/2) of the player to move at the current decision point.</summary>
    public int CurrentPlayerId { get; private set; }

    /// <summary>Winner PlayerId (1/2) once the game is over, else 0.</summary>
    public int Winner { get; private set; }

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
        Winner = 0;
        Observe();
    }

    /// <summary>Apply the current mover's chosen legal action; returns done. Read props after.</summary>
    public bool Step(int actionIndex)
    {
        if (actionIndex < 0 || actionIndex >= _legal.Count)
            actionIndex = 0; // defensive; caller should only pass legal indices

        _game.Process(_legal[actionIndex]);
        _decisions++;

        if (_game.State == State.COMPLETE || _decisions >= MaxDecisions)
        {
            Winner = _game.Player1.PlayState == PlayState.WON ? 1
                   : _game.Player2.PlayState == PlayState.WON ? 2 : 0;
            Tokens = System.Array.Empty<float[]>();
            LegalActionFeatures = System.Array.Empty<float[]>();
            Privileged = new float[PrivDim];
            return true;
        }

        Observe();
        return false;
    }

    private void Observe()
    {
        Controller me = _game.CurrentPlayer;
        CurrentPlayerId = me.PlayerId;
        _legal = me.Options();
        var feats = new float[_legal.Count][];
        for (int i = 0; i < _legal.Count; i++)
            feats[i] = ActionEncoder.Encode(_legal[i], me);
        LegalActionFeatures = feats;
        Privileged = PrivilegedEncoder.Encode(_game.CurrentOpponent);
        Tokens = TokenEncoder.Encode(_game);
    }
}
