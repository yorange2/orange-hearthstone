using SabberStoneCore.Model;
using SabberStoneCore.Model.Entities;

namespace SabberStoneEnv;

/// <summary>
/// "v2" state representation for the entity-transformer policy: instead of one flat 144-float
/// vector, the state is a variable-length SET of entity tokens (hero, board minions, hand
/// cards, weapons, hero power), each an 18-float vector. A transformer attends over the set
/// (permutation-invariant, any board size) — relational reasoning the flat vector can't do.
///
/// Observable-only (learner's perspective): the opponent's hand is hidden, so it is NOT
/// tokenized. This is separate from the flat FeatureExtractor contract (still used by the
/// supervised value-net pipeline); it is the RL policy's trunk input.
///
/// Token layout (Dim = 18):
///   [0..7]  type one-hot: my_hero, opp_hero, my_minion, opp_minion, my_hand, my_weapon,
///           opp_weapon, my_hero_power
///   [8]     mine (belongs to the learner)
///   [9]     cost /10
///   [10]    attack /12
///   [11]    health or durability /12
///   [12..16] taunt, divine shield, windfury, lifesteal, poisonous
///   [17]    can attack now
/// </summary>
public static class TokenEncoder
{
    public const int Dim = 18;

    // token type indices
    private const int MyHero = 0, OppHero = 1, MyMinion = 2, OppMinion = 3,
                      MyHand = 4, MyWeapon = 5, OppWeapon = 6, MyPower = 7;

    /// <summary>
    /// Tokens plus a parallel array of <see cref="CardVocab"/> indices — one per token, same
    /// order. The identity is kept out of the float vector on purpose: it is a categorical
    /// lookup for an embedding, not a magnitude, and stuffing a large integer into a normalized
    /// float feature would be both lossy and meaningless to a linear layer.
    /// </summary>
    public static (float[][] Tokens, int[] CardIds) Encode(Game game) => Encode(game, game.CurrentPlayer);

    /// <summary>
    /// Encode from an explicit perspective. Needed for one-ply lookahead: after processing a
    /// candidate action the turn may flip, and a state must still be valued from the *acting*
    /// player's point of view or the sign of the critic's output is inverted.
    /// </summary>
    public static (float[][] Tokens, int[] CardIds) Encode(Game game, Controller me)
    {
        Controller opp = me.Opponent;
        var toks = new List<float[]>();
        var ids = new List<int>();

        void Add(float[] tok, Card? card)
        {
            toks.Add(tok);
            ids.Add(CardVocab.IndexOf(card));
        }

        Add(HeroTok(MyHero, true, me.Hero), me.Hero.Card);
        Add(HeroTok(OppHero, false, opp.Hero), opp.Hero.Card);
        foreach (Minion m in me.BoardZone) Add(MinionTok(MyMinion, true, m), m.Card);
        foreach (Minion m in opp.BoardZone) Add(MinionTok(OppMinion, false, m), m.Card);
        foreach (IPlayable p in me.HandZone) Add(HandTok(MyHand, true, p), p.Card);
        if (me.Hero.Weapon != null) Add(WeaponTok(MyWeapon, true, me.Hero.Weapon), me.Hero.Weapon.Card);
        if (opp.Hero.Weapon != null) Add(WeaponTok(OppWeapon, false, opp.Hero.Weapon), opp.Hero.Weapon.Card);
        if (me.Hero.HeroPower != null) Add(PowerTok(MyPower, true, me.Hero.HeroPower), me.Hero.HeroPower.Card);

        return (toks.ToArray(), ids.ToArray());
    }

    private static float[] New(int type, bool mine)
    {
        var t = new float[Dim];
        t[type] = 1f;
        t[8] = mine ? 1f : 0f;
        return t;
    }

    private static float[] HeroTok(int type, bool mine, Hero h)
    {
        var t = New(type, mine);
        t[10] = h.AttackDamage / 12f;
        t[11] = (h.Health + h.Armor) / 12f;
        t[17] = h.CanAttack ? 1f : 0f;
        return t;
    }

    private static float[] MinionTok(int type, bool mine, Minion m)
    {
        var t = New(type, mine);
        t[9] = m.Card.Cost / 10f;
        t[10] = m.AttackDamage / 12f;
        t[11] = m.Health / 12f;
        t[12] = m.HasTaunt ? 1f : 0f;
        t[13] = m.HasDivineShield ? 1f : 0f;
        t[14] = m.HasWindfury ? 1f : 0f;
        t[15] = m.HasLifeSteal ? 1f : 0f;
        t[16] = m.Poisonous ? 1f : 0f;
        t[17] = m.CanAttack ? 1f : 0f;
        return t;
    }

    private static float[] HandTok(int type, bool mine, IPlayable p)
    {
        var t = New(type, mine);
        t[9] = p.Cost / 10f;
        if (p is Minion m) { t[10] = m.AttackDamage / 12f; t[11] = m.Health / 12f; }
        return t;
    }

    private static float[] WeaponTok(int type, bool mine, Weapon w)
    {
        var t = New(type, mine);
        t[9] = w.Card.Cost / 10f;
        t[10] = w.AttackDamage / 12f;
        t[11] = w.Durability / 12f;
        return t;
    }

    private static float[] PowerTok(int type, bool mine, HeroPower hp)
    {
        var t = New(type, mine);
        t[9] = hp.Card.Cost / 10f;
        return t;
    }
}
