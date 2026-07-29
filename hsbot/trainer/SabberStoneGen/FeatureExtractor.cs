namespace SabberStoneGen;

/// <summary>
/// Training-side implementation of feature contract v1 (hsbot/docs/FEATURES.md). Must be
/// byte-identical to the Kotlin <c>WarFeatureExtractor</c> and the Python
/// <c>reference_features.py</c>; all three are checked against hsbot/fixtures/*.json.
///
/// Uses float arithmetic with the same divisors as the other implementations so the 1e-6
/// parity tolerance holds.
/// </summary>
public static class FeatureExtractor
{
    public const string Version = "v1";
    public const int Length = 144;

    private const int GlobalPerPlayer = 15;
    private const int TurnBase = 30;
    private const int MinionSectionBase = 32;
    private const int MinionSlots = 7;
    private const int MinionFeats = 8;
    private const int MinionPerPlayer = MinionSlots * MinionFeats; // 56

    public static float[] Extract(ObservableState s)
    {
        var f = new float[Length];

        WriteGlobals(f, 0 * GlobalPerPlayer, s.Me);
        WriteGlobals(f, 1 * GlobalPerPlayer, s.Opp);

        f[TurnBase + 0] = s.Turn / 30f;
        f[TurnBase + 1] = s.MeIsFirst ? 1f : 0f;

        WriteMinions(f, MinionSectionBase + 0 * MinionPerPlayer, s.Me);
        WriteMinions(f, MinionSectionBase + 1 * MinionPerPlayer, s.Opp);

        return f;
    }

    private static void WriteGlobals(float[] f, int b, Side p)
    {
        int sumAttack = 0, sumHealth = 0, taunts = 0, divine = 0;
        foreach (var m in p.Board)
        {
            sumAttack += m.Attack;
            sumHealth += m.Health;
            if (m.Taunt) taunts++;
            if (m.DivineShield) divine++;
        }

        f[b + 0] = p.Hero.EffectiveHp / 30f;
        f[b + 1] = p.Hero.Armor / 30f;
        f[b + 2] = p.Hand / 10f;
        f[b + 3] = p.Deck / 30f;
        f[b + 4] = p.Board.Count / 7f;
        f[b + 5] = sumAttack / 30f;
        f[b + 6] = sumHealth / 40f;
        f[b + 7] = taunts / 7f;
        f[b + 8] = divine / 7f;
        f[b + 9] = p.Secrets / 5f;
        f[b + 10] = (p.Weapon?.Attack ?? 0) / 10f;
        f[b + 11] = (p.Weapon?.Durability ?? 0) / 5f;
        f[b + 12] = p.Mana.Max / 10f;
        f[b + 13] = p.Mana.Overload / 5f;
        f[b + 14] = p.Fatigue / 10f;
    }

    private static void WriteMinions(float[] f, int b, Side p)
    {
        int n = Math.Min(MinionSlots, p.Board.Count);
        for (int i = 0; i < n; i++)
        {
            var m = p.Board[i];
            int o = b + i * MinionFeats;
            f[o + 0] = m.Attack / 12f;
            f[o + 1] = m.Health / 12f;
            f[o + 2] = m.CanAttack ? 1f : 0f;
            f[o + 3] = m.Taunt ? 1f : 0f;
            f[o + 4] = m.DivineShield ? 1f : 0f;
            f[o + 5] = m.Windfury ? 1f : 0f;
            f[o + 6] = m.Lifesteal ? 1f : 0f;
            f[o + 7] = m.Poisonous ? 1f : 0f;
        }
        // remaining slots already zero
    }
}
