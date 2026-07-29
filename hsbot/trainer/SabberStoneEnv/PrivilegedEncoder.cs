using SabberStoneCore.Enums;
using SabberStoneCore.Model.Entities;

namespace SabberStoneEnv;

/// <summary>
/// Encodes the opponent's HIDDEN information (hand contents) as a compact vector used ONLY by
/// the critic during training — the "Cheat" technique from Xiao et al., "Mastering Strategy
/// Card Game (Hearthstone) with Improved Techniques" (IEEE CoG 2023), adapted as an
/// asymmetric (privileged) actor-critic.
///
/// The policy never sees this (it stays observable-only, so it transfers to the live client);
/// the value head does, which lowers value-estimation variance in this POMDP and speeds/
/// stabilises learning. Because our critic is unused at inference, there is no train/test gap.
/// </summary>
public static class PrivilegedEncoder
{
    public const int Dim = 8;

    public static float[] Encode(Controller opp)
    {
        int n = 0, minions = 0, spells = 0, totalCost = 0, maxCost = 0, sumAtk = 0, sumHp = 0, statCount = 0;
        foreach (IPlayable p in opp.HandZone)
        {
            n++;
            totalCost += p.Cost;
            if (p.Cost > maxCost) maxCost = p.Cost;
            CardType ct = p.Card.Type;
            if (ct == CardType.MINION) minions++;
            else if (ct == CardType.SPELL) spells++;
            if (p is Minion m) { sumAtk += m.AttackDamage; sumHp += m.Health; statCount++; }
        }
        float avgAtk = statCount > 0 ? (float)sumAtk / statCount : 0f;
        float avgHp = statCount > 0 ? (float)sumHp / statCount : 0f;

        return new[]
        {
            n / 10f,
            totalCost / 30f,
            minions / 10f,
            spells / 10f,
            maxCost / 10f,
            avgAtk / 12f,
            avgHp / 12f,
            opp.DeckZone.Count / 30f,
        };
    }
}
