using SabberStoneCore.Enums;
using SabberStoneCore.Model.Entities;
using SabberStoneCore.Tasks.PlayerTasks;

namespace SabberStoneEnv;

/// <summary>
/// Encodes a legal <see cref="PlayerTask"/> as a fixed-length feature vector for the RL
/// policy's action-scoring head. The policy scores each legal action's vector and softmaxes
/// over the (variable-size) legal set — so masking of illegal actions is implicit: only legal
/// actions are ever encoded.
///
/// Layout (Dim = 20), all values roughly in [0,1]:
///   [0..6]   task-type one-hot: CHOOSE, CONCEDE, END_TURN, HERO_ATTACK, HERO_POWER,
///            MINION_ATTACK, PLAY_CARD  (index = (int)PlayerTaskType)
///   [7]      has source
///   [8]      source cost /10
///   [9]      source attack /12
///   [10]     source health /12
///   [11..13] source card type: isMinion, isSpell, isWeapon
///   [14]     has target
///   [15]     target is hero
///   [16]     target is friendly (belongs to the acting player)
///   [17]     target attack /12
///   [18]     target health /12
///   [19]     target has taunt
/// </summary>
public static class ActionEncoder
{
    public const int Dim = 20;
    private const int NumTaskTypes = 7; // CHOOSE..PLAY_CARD

    public static float[] Encode(PlayerTask t, Controller me)
    {
        var f = new float[Dim];

        int typeIdx = (int)t.PlayerTaskType;
        if (typeIdx >= 0 && typeIdx < NumTaskTypes) f[typeIdx] = 1f;

        IPlayable? src = t.Source;
        if (src != null)
        {
            f[7] = 1f;
            f[8] = src.Card.Cost / 10f;
            (int atk, int hp) = StatsOf(src);
            f[9] = atk / 12f;
            f[10] = hp / 12f;
            CardType ct = src.Card.Type;
            f[11] = ct == CardType.MINION ? 1f : 0f;
            f[12] = ct == CardType.SPELL ? 1f : 0f;
            f[13] = ct == CardType.WEAPON ? 1f : 0f;
        }

        ICharacter? tgt = t.Target;
        if (tgt != null)
        {
            f[14] = 1f;
            f[15] = tgt is Hero ? 1f : 0f;
            f[16] = ReferenceEquals(tgt.Controller, me) ? 1f : 0f;
            f[17] = tgt.AttackDamage / 12f;
            f[18] = tgt.Health / 12f;
            f[19] = (tgt as Minion)?.HasTaunt == true ? 1f : 0f;
        }

        return f;
    }

    private static (int atk, int hp) StatsOf(IPlayable p) => p switch
    {
        Character c => (c.AttackDamage, c.Health),
        _ => (0, 0), // spells / weapons in hand have no board stats
    };
}
