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
    private readonly DeckMode _deckMode;
    private const int MaxDecisions = 800; // safety cap; HS games terminate via fatigue anyway
    private const double PotentialScale = 200.0;
    private const double BoardPotentialScale = 60.0;

    /// <summary>How each game's decks are chosen.</summary>
    public enum DeckMode
    {
        /// <summary>Mage mirror with a deterministic 30-card fill; only draw order varies. Lowest
        /// variance, and the setting every number in the README so far was measured on. Also the
        /// setting in which card identity is worth least: both players draw from the same 30
        /// cards, so there is nothing unseen to generalise to.</summary>
        Fixed,
        /// <summary>Random class, random legal 30-card deck, **identical for both players**. The
        /// card pool varies game to game (so card semantics matter) while the matchup stays fair
        /// (so win rate still measures policy, not who drew the better pile).</summary>
        VariedMirror,
        /// <summary>Random classes and an independent random fill per player — maximum variety,
        /// but deck-quality asymmetry lands directly in the win rate as noise.</summary>
        Random,
    }

    /// <summary>Which board-state function to use as the shaping potential Φ.</summary>
    public enum PotentialMode
    {
        /// <summary>tanh(MidRangeScore/200) — the greedy opponent's *own* objective. Shaping with
        /// it rewards the agent for maximizing exactly what the heuristic maximizes, which anchors
        /// the policy at heuristic-level play.</summary>
        MidRange,
        /// <summary>Hand-rolled health + board-material differential. Correlated with MidRange (any
        /// sane board eval is) but not identical to greedy's objective.</summary>
        Board,
        /// <summary>No shaping signal; Φ ≡ 0. Isolates the terminal ±1 reward.</summary>
        None,
    }

    private readonly PotentialMode _potentialMode;

    // MidRangeScore reused as the shaping potential Φ (same heuristic the greedy opponent uses).
    private readonly SabberStoneBasicAI.Score.Score _potentialScorer = new SabberStoneBasicAI.Score.MidRangeScore();
    private readonly SabberStoneGen.SabberStoneObserver _observer = new SabberStoneGen.SabberStoneObserver();

    private Game _game = null!;
    private List<SabberStoneCore.Tasks.PlayerTasks.PlayerTask> _legal = new();
    private int _decisions;

    public HearthstoneEnv(int seed, DeckMode deck = DeckMode.Random, PotentialMode potential = PotentialMode.MidRange)
    {
        _rnd = new Random(seed);
        _deckMode = deck;
        _potentialMode = potential;
    }

    /// <summary>Entity tokens for the current decision point: [numTokens, TokenDim].</summary>
    public float[][] Tokens { get; private set; } = System.Array.Empty<float[]>();

    /// <summary>CardVocab index per token, parallel to <see cref="Tokens"/>: [numTokens].</summary>
    public int[] CardIds { get; private set; } = System.Array.Empty<int>();

    /// <summary>Legal-action features: [numActions, ActDim].</summary>
    public float[][] LegalActionFeatures { get; private set; } = System.Array.Empty<float[]>();

    /// <summary>Privileged (opponent-hidden) features for the critic only: [PrivDim].</summary>
    public float[] Privileged { get; private set; } = new float[PrivDim];

    /// <summary>PlayerId (1/2) of the player to move at the current decision point.</summary>
    public int CurrentPlayerId { get; private set; }

    /// <summary>Winner PlayerId (1/2) once the game is over, else 0.</summary>
    public int Winner { get; private set; }

    /// <summary>Action index chosen by the last StepGreedy() call (for imitation learning).</summary>
    public int GreedyActionIndex { get; private set; }

    /// <summary>Normalized board-score potential Φ ∈ [-1,1] for the current mover (reward shaping).</summary>
    public float Potential { get; private set; }

    /// <summary>Flat 144-float observable features for the current mover (FEATURES.md v1, for value-net training data).</summary>
    public float[] Flat { get; private set; } = new float[SabberStoneGen.FeatureExtractor.Length];

    public void Reset()
    {
        CardClass p1, p2;
        List<Card>? deck1 = null, deck2 = null;

        if (_deckMode == DeckMode.VariedMirror)
        {
            (CardClass cls, List<Card> deck) = DeckBuilder.Mirror(Classes, _rnd);
            p1 = p2 = cls;
            // Two copies of the same list: the decks are identical in content, but SabberStone
            // owns each player's list once the game starts, so sharing one instance across both
            // seats would couple them.
            deck1 = new List<Card>(deck);
            deck2 = new List<Card>(deck);
        }
        else
        {
            // Fixed: a Mage mirror with a deterministic 30-card fill (only draw order varies) —
            // big variance reduction vs random classes + random fill.
            p1 = _deckMode == DeckMode.Fixed ? CardClass.MAGE : Classes[_rnd.Next(Classes.Length)];
            p2 = _deckMode == DeckMode.Fixed ? CardClass.MAGE : Classes[_rnd.Next(Classes.Length)];
        }

        _game = new Game(new GameConfig
        {
            // Derive the game's own RNG from the env seed. Without this, GameConfig.RandomSeed is
            // null and SabberStone builds a time-based Random (Game.cs:276), so shuffles/draws —
            // and therefore whole games — differ run to run even at a fixed env seed. That made
            // evaluation unreproducible: identical eval commands on one checkpoint swung 13 points.
            RandomSeed = _rnd.Next(),
            StartPlayer = _rnd.Next(1, 3),
            Player1HeroClass = p1,
            Player2HeroClass = p2,
            Player1Deck = deck1,
            Player2Deck = deck2,
            // VariedMirror supplies a complete 30-card list, so fill has nothing left to add;
            // leaving it on would top up from a different pool and break the mirror.
            FillDecks = _deckMode != DeckMode.VariedMirror,
            FillDecksPredictably = _deckMode == DeckMode.Fixed,
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

    /// <summary>Step by playing the current mover's greedy (MidRangeScore) heuristic action.</summary>
    public bool StepGreedy() => StepGreedy("midrange");

    /// <summary>Step by playing the current mover's greedy action using the named strategy
    /// (aggro, control, fatigue, ramp, midrange).</summary>
    public bool StepGreedy(string strategyName)
    {
        var s = GreedyOpponent.Parse(strategyName);
        GreedyActionIndex = GreedyOpponent.BestAction(_game, s);
        return Step(GreedyActionIndex);
    }

    /// <summary>One resulting state per legal action, for critic-scored one-ply lookahead.</summary>
    public sealed class SimState
    {
        public float[][] Tokens { get; set; } = System.Array.Empty<float[]>();
        public int[] CardIds { get; set; } = System.Array.Empty<int>();
        public float[] Priv { get; set; } = System.Array.Empty<float>();
        public bool Terminal { get; set; }
        /// <summary>True when the resulting state's player-to-move is still the acting player.
        /// When false the critic's value is from the opponent's view and must be negated.</summary>
        public bool Mine { get; set; }
        /// <summary>+1/-1/0 from the acting player's perspective when Terminal.</summary>
        public int Outcome { get; set; }
    }

    /// <summary>
    /// Phase 3: clone-and-apply each legal action, returning the resulting state encoded from the
    /// *current mover's* perspective. This is the same one-ply lookahead the greedy heuristic
    /// gets; the difference is that the caller scores the results with the learned critic instead
    /// of a handcrafted board score.
    ///
    /// Clones inherit the parent RNG stream (resetRandomSeed: false) so every option is evaluated
    /// under the same draw and the comparison stays deterministic.
    /// </summary>
    public SimState[] SimulateAll()
    {
        int actingPid = _game.CurrentPlayer.PlayerId;
        var outp = new SimState[_legal.Count];
        for (int i = 0; i < _legal.Count; i++)
        {
            Game clone = _game.Clone(resetRandomSeed: false);
            var opts = clone.CurrentPlayer.Options();
            if (i >= opts.Count) { outp[i] = new SimState(); continue; }
            clone.Process(opts[i]);

            if (clone.State == State.COMPLETE)
            {
                int winner = clone.Player1.PlayState == PlayState.WON ? 1
                           : clone.Player2.PlayState == PlayState.WON ? 2 : 0;
                outp[i] = new SimState
                {
                    Terminal = true,
                    Outcome = winner == 0 ? 0 : (winner == actingPid ? 1 : -1),
                };
                continue;
            }

            // Encode from the player *to move*, which is the only perspective the critic was ever
            // trained on. Encoding from the acting player's view after the turn flips is
            // out-of-distribution and the value estimate becomes meaningless (measured: it drove
            // win rate from 0.667 to 0.067). `Mine` tells the caller whether to negate — the
            // standard negamax convention.
            Controller mover = clone.CurrentPlayer;
            var (tok, ids) = TokenEncoder.Encode(clone, mover);
            outp[i] = new SimState
            {
                Tokens = tok, CardIds = ids, Priv = PrivilegedEncoder.Encode(mover.Opponent),
                Mine = mover.PlayerId == actingPid,
            };
        }
        return outp;
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
        (Tokens, CardIds) = TokenEncoder.Encode(_game);
        Flat = SabberStoneGen.FeatureExtractor.Extract(_observer.Observe(_game));
        Potential = ComputePotential(me);
    }

    /// <summary>Shaping potential Φ ∈ [-1,1] for the given controller, per <see cref="PotentialMode"/>.</summary>
    private float ComputePotential(Controller me)
    {
        switch (_potentialMode)
        {
            case PotentialMode.None:
                return 0f;

            case PotentialMode.Board:
            {
                // Health/armor differential plus board material (attack + health of minions),
                // deliberately NOT routed through any SabberStoneBasicAI scorer.
                Controller opp = me.Opponent;
                int hp = (me.Hero.Health + me.Hero.Armor) - (opp.Hero.Health + opp.Hero.Armor);
                int material = 0;
                foreach (Minion m in me.BoardZone) material += m.AttackDamage + m.Health;
                foreach (Minion m in opp.BoardZone) material -= m.AttackDamage + m.Health;
                return (float)System.Math.Tanh((hp + 2 * material) / BoardPotentialScale);
            }

            default:
                _potentialScorer.Controller = me;
                return (float)System.Math.Tanh(_potentialScorer.Rate() / PotentialScale);
        }
    }
}
